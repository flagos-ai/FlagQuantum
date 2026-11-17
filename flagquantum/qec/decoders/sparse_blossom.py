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

    def decode_batch(self, detection_events: torch.Tensor) -> torch.Tensor:
        """Return an owned CPU bool matrix of logical predictions in row order."""
        if not isinstance(detection_events, torch.Tensor):
            raise TypeError("detection_events must be a torch.Tensor")
        if (
            detection_events.layout != torch.strided
            or detection_events.dtype not in _BINARY_DTYPES
        ):
            raise TypeError("detection_events must be a dense binary tensor")
        if detection_events.device.type != "cpu":
            raise CapabilityError("Sparse Blossom accepts CPU tensors only")
        if (
            detection_events.ndim != 2
            or detection_events.shape[1] != self._graph.num_detectors
        ):
            raise ValueError("detection_events must have one column per graph detector")
        if not bool(((detection_events == 0) | (detection_events == 1)).all()):
            raise ValueError("detection_events must contain only binary values")
        shape = (detection_events.shape[0], self._graph.num_observables)
        if not shape[0]:
            return torch.empty(shape, dtype=torch.bool, device="cpu")
        contiguous = detection_events.to(torch.uint8).contiguous()
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
