"""Optional SDK execution for vendor-neutral weighted matching graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.decoders.matching import _defects
from flagquantum.qec.decoders.matching_graph import MatchingGraph, MatchingSolution

from . import pymatching as _pymatching_adapter


def _build_matching(graph: MatchingGraph, *, record_edge_ids: bool = False) -> Any:
    checks, faults, weights = graph.to_check_matrices()
    if record_edge_ids:
        faults = torch.cat(
            (faults, torch.eye(len(graph.edges), dtype=torch.uint8)), dim=0
        )
    matching = _pymatching_adapter._pymatching().Matching.from_check_matrix(
        checks.numpy(),
        weights=weights.numpy(),
        faults_matrix=faults.numpy(),
        merge_strategy="disallow",
        use_virtual_boundary_node=True,
    )
    matching.set_boundary_nodes(set(graph.boundary_nodes))
    matching.ensure_num_fault_ids(
        graph.num_fault_ids + (len(graph.edges) if record_edge_ids else 0)
    )
    if matching.num_nodes != graph.num_nodes or matching.num_edges != len(graph.edges):
        raise CapabilityError(
            "SDK graph preparation changed the declared node or edge shape"
        )
    return matching


@dataclass(frozen=True)
class PyMatchingGraphDecoder:
    """Prepare SDK matching without interpreting edge costs as probabilities.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
        >>> if find_spec("pymatching") is not None:
        ...     graph = MatchingGraph(1, 1, (MatchingEdge((0,), -1.0, (0,)),))
        ...     assert PyMatchingGraphDecoder(graph).decode((0,)).weight == -1.0
    """

    graph: MatchingGraph
    _matching: Any = field(init=False, repr=False, compare=False)
    _record_matching: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, MatchingGraph):
            raise TypeError("graph must be a MatchingGraph")
        object.__setattr__(self, "_matching", _build_matching(self.graph))
        object.__setattr__(
            self, "_record_matching", _build_matching(self.graph, record_edge_ids=True)
        )

    def _events(self, detection_events: Iterable[int]) -> tuple[int, ...]:
        events = _defects(detection_events, num_detectors=self.graph.num_nodes)
        if set(events).intersection(self.graph.boundary_nodes):
            raise ValueError("boundary nodes cannot be detection events")
        return events

    def decode(self, detection_events: Iterable[int]) -> MatchingSolution:
        """Return one coherent SDK solution with edge indices and both cost readings.

        Empty syndromes still execute because negative-cost cycles may be selected.
        Unsupported syndromes or malformed SDK results raise CapabilityError.

        Examples:
            >>> from importlib.util import find_spec
            >>> from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
            >>> if find_spec("pymatching") is not None:
            ...     graph = MatchingGraph(1, 1, (MatchingEdge((0,), 2.0, (0,)),))
            ...     assert PyMatchingGraphDecoder(graph).decode((0,)).edge_indices == (0,)
        """
        events = self._events(detection_events)
        syndrome = [int(i in events) for i in range(self.graph.num_nodes)]
        try:
            raw = self._record_matching.decode(syndrome, return_weight=True)
        except ValueError as exc:
            raise CapabilityError(f"SDK matching failed: {exc}") from exc
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise CapabilityError("SDK returned an invalid solution pair")
        bits = _bits(raw[0], (self.graph.num_fault_ids + len(self.graph.edges),))
        cost = raw[1]
        if (
            isinstance(cost, bool)
            or not isinstance(cost, Real)
            or not math.isfinite(float(cost))
        ):
            raise CapabilityError("SDK returned an invalid solution weight")
        faults = tuple(
            int(i) for i in bits[: self.graph.num_fault_ids].nonzero().flatten()
        )
        indices = tuple(
            int(i) for i in bits[self.graph.num_fault_ids :].nonzero().flatten()
        )
        expected_faults: set[int] = set()
        expected_events: set[int] = set()
        for index in indices:
            edge = self.graph.edges[index]
            expected_faults.symmetric_difference_update(edge.fault_ids)
            expected_events.symmetric_difference_update(edge.detectors)
        expected_events.difference_update(self.graph.boundary_nodes)
        if expected_faults != set(faults) or expected_events != set(events):
            raise CapabilityError(
                "SDK solution labels disagree with the selected edge parity"
            )
        return MatchingSolution(
            faults,
            indices,
            sum(self.graph.edges[i].weight for i in indices),
            float(cost),
        )

    def decode_batch(
        self,
        detection_events: torch.Tensor,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return owned CPU logical predictions and signed SDK weights.

        Dense input has one column per declared node; boundary columns are zero.
        Packed bytes use LSB-first order and canonical zero padding.

        Examples:
            >>> import torch
            >>> from importlib.util import find_spec
            >>> from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
            >>> if find_spec("pymatching") is not None:
            ...     graph = MatchingGraph(1, 1, (MatchingEdge((0,), 2.0, (0,)),))
            ...     bits, costs = PyMatchingGraphDecoder(graph).decode_batch(torch.tensor([[1]]))
            ...     assert bits.tolist() == [[True]]
        """
        if (
            type(bit_packed_shots) is not bool
            or type(bit_packed_predictions) is not bool
        ):
            raise TypeError("packed flags must be bool")
        if (
            not isinstance(detection_events, torch.Tensor)
            or detection_events.layout != torch.strided
        ):
            raise TypeError("detection_events must be a dense tensor")
        if detection_events.device.type != "cpu":
            raise CapabilityError("matching batch requires CPU input")
        if detection_events.dtype not in (
            torch.bool,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("detection_events must have binary bool or integer dtype")
        width = (
            (self.graph.num_nodes + 7) // 8
            if bit_packed_shots
            else self.graph.num_nodes
        )
        if detection_events.ndim != 2 or detection_events.shape[1] != width:
            raise ValueError("detection_events has the wrong declared width")
        if bit_packed_shots:
            if detection_events.dtype != torch.uint8:
                raise TypeError("packed input must be uint8")
            remainder = self.graph.num_nodes % 8
            if remainder and bool((detection_events[:, -1] >> remainder).any()):
                raise ValueError("packed node padding must be zero")
        elif not bool(((detection_events == 0) | (detection_events == 1)).all()):
            raise ValueError("detection_events must be binary")
        for node in self.graph.boundary_nodes:
            active = (
                (detection_events[:, node // 8] >> (node % 8)) & 1
                if bit_packed_shots
                else detection_events[:, node]
            )
            if bool(active.any()):
                raise ValueError("boundary columns must be zero")
        rows = detection_events.shape[0]
        columns = (
            (self.graph.num_fault_ids + 7) // 8
            if bit_packed_predictions
            else self.graph.num_fault_ids
        )
        if rows == 0:
            return torch.empty(
                (0, columns),
                dtype=torch.uint8 if bit_packed_predictions else torch.bool,
            ), torch.empty(0, dtype=torch.float64)
        owned = detection_events.to(dtype=torch.uint8, copy=True).contiguous()
        try:
            raw = self._matching.decode_batch(
                owned.numpy(),
                return_weights=True,
                bit_packed_shots=bit_packed_shots,
                bit_packed_predictions=bit_packed_predictions,
            )
        except ValueError as exc:
            raise CapabilityError(f"SDK batch matching failed: {exc}") from exc
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise CapabilityError("SDK returned an invalid batch pair")
        predictions = _bits(raw[0], (rows, columns), packed=bit_packed_predictions)
        if bit_packed_predictions:
            remainder = self.graph.num_fault_ids % 8
            if remainder and bool((predictions[:, -1] >> remainder).any()):
                raise CapabilityError("SDK returned nonzero fault padding")
        try:
            weights = torch.as_tensor(raw[1])
        except (ValueError, TypeError, RuntimeError) as exc:
            raise CapabilityError("SDK returned an invalid weight buffer") from exc
        if (
            weights.device.type != "cpu"
            or weights.layout != torch.strided
            or weights.dtype != torch.float64
            or weights.shape != (rows,)
            or not bool(torch.isfinite(weights).all())
        ):
            raise CapabilityError("SDK returned invalid batch weights")
        return (
            predictions.to(
                dtype=torch.uint8 if bit_packed_predictions else torch.bool
            ).clone(),
            weights.clone(),
        )

    def decode_to_matched_detectors(
        self, detection_events: Iterable[int]
    ) -> tuple[tuple[int, int | None], ...]:
        """Return endpoint pairs for nonnegative graphs only.

        Examples:
            >>> from importlib.util import find_spec
            >>> from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
            >>> if find_spec("pymatching") is not None:
            ...     graph = MatchingGraph(1, 0, (MatchingEdge((0,), 1.0),))
            ...     assert PyMatchingGraphDecoder(graph).decode_to_matched_detectors((0,)) == ((0, None),)
        """
        if any(edge.weight < 0 for edge in self.graph.edges):
            raise CapabilityError("matched endpoints require nonnegative edge weights")
        events = self._events(detection_events)
        if not events:
            return ()
        try:
            raw = self._matching.decode_to_matched_dets_array(
                [int(i in events) for i in range(self.graph.num_nodes)]
            )
        except ValueError as exc:
            raise CapabilityError(f"SDK endpoint matching failed: {exc}") from exc
        try:
            pairs = torch.as_tensor(raw)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise CapabilityError("SDK returned an invalid endpoint buffer") from exc
        if (
            pairs.device.type != "cpu"
            or pairs.layout != torch.strided
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or pairs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise CapabilityError("SDK returned invalid endpoint shape or dtype")
        active, seen = set(events), set()
        result: list[tuple[int, int | None]] = []
        for pair in pairs:
            first, second = int(pair[0]), int(pair[1])
            if first not in active or first in seen:
                raise CapabilityError("SDK returned an inactive or repeated endpoint")
            seen.add(first)
            if second == -1 or second in self.graph.boundary_nodes:
                result.append((first, None))
            else:
                if second not in active or second in seen:
                    raise CapabilityError(
                        "SDK returned an inactive or repeated endpoint"
                    )
                seen.add(second)
                result.append((min(first, second), max(first, second)))
        if seen != active:
            raise CapabilityError("SDK omitted active endpoints")
        return tuple(sorted(result, key=lambda pair: pair[0]))


def _bits(raw: object, shape: tuple[int, ...], *, packed: bool = False) -> torch.Tensor:
    try:
        result = torch.as_tensor(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CapabilityError("SDK returned an invalid prediction buffer") from exc
    if (
        result.device.type != "cpu"
        or result.layout != torch.strided
        or result.dtype != torch.uint8
        or tuple(result.shape) != shape
    ):
        raise CapabilityError("SDK returned invalid prediction shape or dtype")
    if not packed and not bool(((result == 0) | (result == 1)).all()):
        raise CapabilityError("SDK returned nonbinary predictions")
    return result
