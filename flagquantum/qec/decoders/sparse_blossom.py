"""FlagQuantum wrapper for the bundled PyMatching Sparse Blossom core."""

from __future__ import annotations

import ctypes
from importlib import import_module

import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.dem.graph import DecodingGraph

_BINARY_DTYPES = frozenset(
    (
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
)


class SparseBlossomDecoder:
    """Predict logical flips on a graph of independent matching mechanisms.

    The compiled solver uses the fixed-point edge weights of PyMatching 2.4.
    A graph can declare at most 64 logical observables and one edge per detector
    pair. The solver reports a capability error if no matching exists.
    """

    def __init__(self, graph: DecodingGraph) -> None:
        if not isinstance(graph, DecodingGraph):
            raise TypeError("Sparse Blossom requires a DecodingGraph")
        if not 1 <= graph.num_observables <= 64:
            raise CapabilityError("Sparse Blossom requires 1 to 64 logical observables")
        if len({edge.detectors for edge in graph.edges}) != len(graph.edges):
            raise CapabilityError("Sparse Blossom requires one edge per detector pair")
        try:
            extension = import_module("flagquantum.qec.decoders._sparse_blossom")
        except ImportError as error:
            raise CapabilityError(
                "The bundled Sparse Blossom extension could not load. "
                "Install a FlagQuantum wheel or build its native extensions."
            ) from error
        self._graph = graph
        self._matcher = extension.create(
            graph.num_detectors,
            graph.num_observables,
            tuple(
                (*edge.detectors, edge.weight, edge.observables) for edge in graph.edges
            ),
        )
        self._decode = extension.decode_batch
        self._decode_packed = extension.decode_packed_batch
        self._decode_weighted = extension.decode_batch_with_weights

    def _input(self, detection_events: torch.Tensor, *, packed: bool) -> torch.Tensor:
        if not isinstance(detection_events, torch.Tensor):
            raise TypeError("detection_events must be a torch.Tensor")
        if packed:
            if (
                detection_events.layout != torch.strided
                or detection_events.dtype != torch.uint8
            ):
                raise TypeError("packed detection_events must be a dense uint8 tensor")
        elif (
            detection_events.layout != torch.strided
            or detection_events.dtype not in _BINARY_DTYPES
        ):
            raise TypeError("detection_events must be a dense binary tensor")
        if detection_events.device.type != "cpu":
            raise CapabilityError("Sparse Blossom accepts CPU tensors only")
        width = (
            (self._graph.num_detectors + 7) // 8
            if packed
            else self._graph.num_detectors
        )
        if detection_events.ndim != 2 or detection_events.shape[1] != width:
            if packed:
                raise ValueError(
                    f"packed detection_events must have shape (shots, {width})"
                )
            raise ValueError("detection_events must have one column per graph detector")
        if packed:
            remainder = self._graph.num_detectors % 8
            if remainder and bool((detection_events[:, -1] >> remainder).any()):
                raise ValueError("unused packed detector padding bits must be zero")
        elif not bool(((detection_events == 0) | (detection_events == 1)).all()):
            raise ValueError("detection_events must contain only binary values")
        return detection_events.to(torch.uint8).contiguous()

    def decode_batch(self, detection_events: torch.Tensor) -> torch.Tensor:
        """Return an owned CPU bool matrix of logical predictions in row order."""
        contiguous = self._input(detection_events, packed=False)
        shape = (contiguous.shape[0], self._graph.num_observables)
        if not shape[0]:
            return torch.empty(shape, dtype=torch.bool, device="cpu")
        dense = ctypes.string_at(contiguous.data_ptr(), contiguous.numel())
        try:
            predictions = self._decode(self._matcher, dense)
        except ValueError as error:
            raise CapabilityError(
                f"Sparse Blossom could not decode the batch: {error}"
            ) from error
        return (
            torch.frombuffer(bytearray(predictions), dtype=torch.uint8)
            .view(shape)
            .bool()
        )

    def decode_packed_batch(self, detection_events: torch.Tensor) -> torch.Tensor:
        """Return owned CPU uint8 logical bytes from LSB-first packed syndromes.

        Each row contains one byte per eight detectors. Unused detector padding
        bits must be zero. Output rows contain one byte per eight observables,
        with zero padding bits. Noncontiguous input is accepted.
        """
        contiguous = self._input(detection_events, packed=True)
        shape = (contiguous.shape[0], (self._graph.num_observables + 7) // 8)
        if not shape[0]:
            return torch.empty(shape, dtype=torch.uint8, device="cpu")
        packed = ctypes.string_at(contiguous.data_ptr(), contiguous.numel())
        try:
            predictions = self._decode_packed(self._matcher, packed)
        except ValueError as error:
            raise CapabilityError(
                f"Sparse Blossom could not decode the packed batch: {error}"
            ) from error
        return torch.frombuffer(bytearray(predictions), dtype=torch.uint8).view(shape)

    def decode_batch_with_weights(
        self,
        detection_events: torch.Tensor,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical predictions and rescaled quantized matching costs.

        Dense input follows decode_batch. Packed input and output follow the
        LSB-first layout of decode_packed_batch. Input/output packing flags are
        independent. Weights are owned CPU float64 values with shape (shots,).
        Both outputs own their buffers. A failed batch returns no partial result.

        The solver doubles rounded fixed-point edge weights. Returned costs
        divide the integer total by the graph's normalization factor, matching
        PyMatching's return_weights convention.

        Raises:
            TypeError: Flags, input type, layout or dtype are invalid.
            ValueError: Input shape, binary values or padding are invalid.
            CapabilityError: Input is not on CPU or matching fails.
        """
        if (
            type(bit_packed_shots) is not bool
            or type(bit_packed_predictions) is not bool
        ):
            raise TypeError("packed flags must be bool")
        contiguous = self._input(detection_events, packed=bit_packed_shots)
        shots = contiguous.shape[0]
        columns = (
            (self._graph.num_observables + 7) // 8
            if bit_packed_predictions
            else self._graph.num_observables
        )
        shape = (shots, columns)
        dtype = torch.uint8 if bit_packed_predictions else torch.bool
        if shots == 0:
            return (
                torch.empty(shape, dtype=dtype, device="cpu"),
                torch.empty(0, dtype=torch.float64, device="cpu"),
            )
        syndrome = ctypes.string_at(contiguous.data_ptr(), contiguous.numel())
        try:
            predictions, costs = self._decode_weighted(
                self._matcher, syndrome, bit_packed_shots, bit_packed_predictions
            )
        except ValueError as error:
            raise CapabilityError(
                f"Sparse Blossom could not decode the weighted batch: {error}"
            ) from error
        bits = torch.frombuffer(bytearray(predictions), dtype=torch.uint8).view(shape)
        weights = torch.frombuffer(bytearray(costs), dtype=torch.float64)
        return bits.to(dtype), weights
