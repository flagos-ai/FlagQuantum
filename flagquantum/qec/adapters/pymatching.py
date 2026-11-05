"""Cross-check the self-implemented matcher against PyMatching.

:mod:`flagquantum.qec.decoders.matching` is the authority for a decoded syndrome, and its
correctness evidence is deliberately independent of the implementation: brute
force over the syndromes small enough to enumerate, and the exhaustive
enumeration of a model's own mechanism distribution. Neither of those is
independent of the detector error model both read, so the one check that is
independent of both is a second implementation that reads the same detector
error model and nothing else.

PyMatching is that second implementation. It is Apache-2.0, it is the reference
minimum-weight perfect matching decoder of the detector error model ecosystem,
and it is not a dependency of this package: it arrives behind the `pymatching`
extra, this module imports it lazily, and importing :mod:`flagquantum.qec`
therefore never loads it. The authority stays in-tree, which is why the extra is
a cross-check and not the decoder.

What the comparison can and cannot establish is stated here rather than implied.
A detector error model does not always have one cheapest explanation. Two
explanations of the same syndrome can have equal weight and disagree about the
logical observables, and where that happens no tie-breaking rule is more correct
than another: the two decoders agree on the *weight* of the cheapest explanation
of every syndrome, and they agree on the *observables* only where the cheapest
explanation is unique. Agreement of the predictions is therefore not asserted
unconditionally, and a disagreement is evidence against neither implementation
until its weight is compared.

That comparison carries a tolerance, and the tolerance is not a convenience.
PyMatching's arithmetic is narrower than this package's: the mechanism of
probability ``0.0198`` that the rotated surface code's first detector has to its
boundary has weight ``3.9020746947749574`` here and ``3.9020747171643912`` in the
weight PyMatching minimizes and reports, a relative difference of ``5.8e-9``.
That difference is neither this package's value rounded to single precision
(``3.9020748138427734``) nor the single-precision probability put back through
``log((1 - p) / p)`` (``3.902074700303017``), so it is internal to PyMatching and
this module does not model it. What follows does not depend on the mechanism: the
graph PyMatching minimizes over is not exactly this one, so two explanations
whose true weights differ by less than that difference can be ordered
differently on the two sides, and an exact comparison of the two selections would
report that as a disagreement between the decoders rather than as the precision
difference it is. A caller therefore compares the two records with a tolerance --
``num_edges * 2**-23 * max(1, weight)``, one single-precision rounding per
selected mechanism, which is twenty times the difference measured above -- and
treats a pair inside it as tied and uncomparable, the same conclusion the
exact-equality case reaches. The record this module returns states the weight as
the exact sum over the mechanisms PyMatching selected, in double precision, so
that the two records hold the same quantity and the comparison measures the two
selections rather than the two summations; ``decode(return_weight=True)`` would
report the narrower one instead.

The translation is from a decoding graph and not from detector error model text
or from a stim circuit, so the cross-check reads exactly what the authority reads
and the model's own construction is not re-measured. Two graphs are refused
rather than approximated, both for the same reason -- a translation that is not
faithful would make the comparison measure the translation instead of the
decoder:

* A detector pair carrying two mechanisms. PyMatching represents a decoding graph
  as a graph with one edge per detector pair, and merging two mechanisms that
  disagree about their logical labels would either lose a flip or invent a weight
  neither mechanism has. The self-implemented matcher can keep both, so this is a
  narrower domain and not a wider one.
* A detector that no mechanism flips. PyMatching infers its detector count from
  the edges it is handed, so such a detector would change the length of the
  syndrome vector and every detector index after it would mean something else.
  A model that declares one is refused instead, because a detector no mechanism
  flips cannot fire.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Any

import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.codes._compat import _retain_module
from flagquantum.qec.decoders.matching import MatchingDecodeResult, _defects
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.dem.graph import DecodingGraph, DecodingGraphEdge

_EXTRA_NAME = "pymatching"
# PyMatching reports the step from a detector to its virtual boundary as an edge
# to this node, which no detector can be.
_BOUNDARY = -1


class MatchingDependencyError(CapabilityError, ImportError):
    """Raised only when the cross-check is requested without PyMatching."""


def _pymatching() -> ModuleType:
    """Import the cross-check implementation, or name the extra that provides it."""

    try:
        return import_module(_EXTRA_NAME)
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise MatchingDependencyError(
            "the matching cross-check requires the optional dependency; install it "
            f"with `pip install 'flagquantum[{_EXTRA_NAME}]'`."
        ) from exc


def _translate(
    graph: DecodingGraph,
    *,
    record_edge_ids: bool = False,
) -> tuple[Any, dict[tuple[int, int], DecodingGraphEdge]]:
    """Build PyMatching's matching graph for one decoding graph.

    Returns the matching graph and the map from a node pair back to the decoding
    graph edge it stands for, which is how a selected edge is read as the
    mechanism this package selected.

    ``record_edge_ids`` appends private edge labels above the logical range so
    one SDK decode can return a coherent full record. The default translation
    and its logical-only batch shape remain unchanged.
    """

    carried: dict[tuple[int, int], DecodingGraphEdge] = {}
    touched: set[int] = set()
    for edge in graph.edges:
        touched.update(edge.detectors)
        existing = carried.get(edge.detectors)
        if existing is not None:
            raise CapabilityError(
                "PyMatching holds one edge per detector pair, and detector pair "
                f"{edge.detectors} carries a mechanism of probability "
                f"{existing.probability!r} and one of probability "
                f"{edge.probability!r}; merging them would either lose a logical "
                "flip or invent a weight neither mechanism has, so this model "
                "cannot be translated faithfully"
            )
        carried[edge.detectors] = edge

    untouched = [
        detector for detector in range(graph.num_detectors) if detector not in touched
    ]
    if untouched:
        raise CapabilityError(
            f"detector D{untouched[0]} is flipped by no mechanism, and PyMatching "
            "infers the detector count from the edges it is given, so the "
            "syndrome vector would not be this graph's"
        )

    matching = _pymatching().Matching()
    for index, edge in enumerate(graph.edges):
        first, second = edge.detectors
        labels = set(edge.observables)
        if record_edge_ids:
            labels.add(graph.num_observables + index)
        if second == graph.boundary_node:
            matching.add_boundary_edge(
                first,
                fault_ids=labels,
                weight=edge.weight,
                error_probability=edge.probability,
            )
        else:
            matching.add_edge(
                first,
                second,
                fault_ids=labels,
                weight=edge.weight,
                error_probability=edge.probability,
            )
    matching.ensure_num_fault_ids(
        graph.num_observables + (graph.num_edges if record_edge_ids else 0)
    )
    if matching.num_detectors != graph.num_detectors:
        raise CapabilityError(
            "PyMatching inferred "
            f"{matching.num_detectors} detectors from this graph's {graph.num_edges} "
            f"mechanisms, and the graph states {graph.num_detectors}; refusing "
            "rather than renumbering a syndrome"
        )
    return matching, carried


def _check_batch_input(detection_events: torch.Tensor, detectors: int) -> None:
    if not isinstance(detection_events, torch.Tensor):
        raise TypeError("detection_events must be a torch.Tensor")
    if detection_events.layout != torch.strided:
        raise TypeError("detection_events must be a dense strided tensor")
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
        raise TypeError("detection_events must have bool or integer dtype")
    if detection_events.device.type != "cpu":
        raise CapabilityError("batch matching accepts CPU tensors only")
    if detection_events.ndim != 2:
        raise ValueError("detection_events must have shape (shots, detectors)")
    if detection_events.shape[1] != detectors:
        raise ValueError(f"detection_events must have {detectors} detector columns")
    if not bool(((detection_events == 0) | (detection_events == 1)).all()):
        raise ValueError("detection_events must contain only binary values 0 and 1")


@dataclass(frozen=True)
class PyMatchingDecoder:
    """Decode a syndrome with PyMatching, to cross-check the in-tree matcher.

    The class mirrors :class:`~flagquantum.qec.MinimumWeightMatchingDecoder`: it
    is built from the same decoding graph or the same detector error model, it
    takes the same detection events, and it returns the same
    :class:`~flagquantum.qec.MatchingDecodeResult`, so the two are compared on one
    syndrome at a time without a translation in between.

    The result's ``observables`` are PyMatching's own prediction rather than a
    prediction this package recomputes from the selected mechanisms, which is
    what keeps the label bookkeeping an independent side of the comparison. Its
    ``weight`` is the exact sum of the selected mechanisms' weights, and not
    PyMatching's single-precision accumulator.

    Full records use an additional temporary graph with private edge labels.
    Logical predictions and edges then come from one SDK solution, including
    ties. This adds graph construction and metadata cost per full-record call;
    it is not a performance optimization. The stored logical-only graph and
    decoder instance fields remain unchanged.
    """

    graph: DecodingGraph
    _matching: Any = field(init=False, repr=False, compare=False)
    _edges: dict[tuple[int, int], DecodingGraphEdge] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.graph, DecodingGraph):
            raise TypeError("graph must be a DecodingGraph")
        matching, edges = _translate(self.graph)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_edges", edges)

    @classmethod
    def from_detector_error_model(cls, model: DetectorErrorModel) -> PyMatchingDecoder:
        """Derive the cross-check decoder of the graph a detector error model defines.

        Args:
            model: The model whose mechanisms become PyMatching's edges.

        Returns:
            The cross-check decoder over the model's detectors.

        Raises:
            CapabilityError: The model is not graphlike, or its graph is outside
                the domain a faithful translation covers.
            MatchingDependencyError: PyMatching is not installed.
        """

        if not isinstance(model, DetectorErrorModel):
            raise TypeError("model must be a DetectorErrorModel")
        return cls(DecodingGraph.from_detector_error_model(model))

    @classmethod
    def from_decoding_graph(cls, graph: DecodingGraph) -> PyMatchingDecoder:
        """Derive the cross-check decoder of a graph a caller already holds.

        The translation is the same one the model route performs after deriving
        the graph, and it is performed here rather than in the registry so that the
        narrowness of the translation's domain -- a detector pair carrying two
        mechanisms and a detector no mechanism flips are both refused -- stays a
        property of this class.

        Raises:
            CapabilityError: The graph is outside the domain a faithful translation
                covers.
            MatchingDependencyError: PyMatching is not installed.
        """

        if not isinstance(graph, DecodingGraph):
            raise TypeError("graph must be a DecodingGraph")
        return cls(graph)

    def _pair(self, first: int, second: int) -> tuple[int, int]:
        """Read one PyMatching edge as the node pair this graph records."""

        if first == _BOUNDARY or second == _BOUNDARY:
            node = second if first == _BOUNDARY else first
            return (node, self.graph.boundary_node)
        return (first, second) if first < second else (second, first)

    def decode_batch(self, detection_events: torch.Tensor) -> torch.Tensor:
        """Predict logical flips for a CPU binary matrix of detection events.

        Args:
            detection_events: Dense bool or integer tensor of shape
                ``(shots, graph.num_detectors)``. Noncontiguous input is copied.

        Returns:
            Caller-owned CPU bool tensor of shape ``(shots, num_observables)``.
            This logical-only path reuses the graph and does not recover edges.

        Raises:
            TypeError: The input is not a dense bool or integer tensor.
            ValueError: The rank, detector width or binary values are invalid.
            CapabilityError: The input is not on CPU, the batch cannot be
                decoded, or the SDK returns an invalid logical prediction.

        Examples:
            Install the optional pymatching extra to execute the matching step.

            >>> from importlib.util import find_spec
            >>> import torch
            >>> from flagquantum.qec import DetectorErrorModel, DemError
            >>> if find_spec("pymatching") is not None:
            ...     model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
            ...     decoder = PyMatchingDecoder.from_detector_error_model(model)
            ...     result = decoder.decode_batch(torch.tensor([[0], [1]]))
            ...     assert result.tolist() == [[False], [True]]
        """
        _check_batch_input(detection_events, self.graph.num_detectors)
        shots = detection_events.shape[0]
        if shots == 0:
            return torch.empty((0, self.graph.num_observables), dtype=torch.bool)
        # Own the SDK buffer even when the caller already supplied uint8.
        syndrome = detection_events.to(dtype=torch.uint8, copy=True).contiguous()
        try:
            predicted = self._matching.decode_batch(syndrome.numpy())
        except ValueError as exc:
            raise CapabilityError(
                f"PyMatching could not decode the batch: {exc}"
            ) from exc
        try:
            result = torch.as_tensor(predicted)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CapabilityError(
                "PyMatching returned an invalid prediction buffer"
            ) from exc
        if result.shape != (shots, self.graph.num_observables):
            raise CapabilityError("PyMatching returned an invalid prediction shape")
        if not bool(((result == 0) | (result == 1)).all()):
            raise CapabilityError("PyMatching returned nonbinary predictions")
        return result.to(dtype=torch.bool, device="cpu").clone()

    def decode_batch_with_weights(
        self,
        detection_events: torch.Tensor,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical predictions and SDK matching weights from one batch call.

        Dense input uses the existing binary CPU contract. Packed input uses
        CPU uint8 bytes, least-significant bit first, with zero unused padding.
        Both outputs own their storage. SDK weights are CPU float64, not the
        recomputed selected-edge sums in complete matching records.

        Raises:
            TypeError: Flags or input carrier/dtype/layout are invalid.
            ValueError: Input shape, values or packed padding is invalid.
            CapabilityError: Input device, SDK decoding or returned buffers
                violate the contract. No partial result is returned.

        Examples:
            >>> from importlib.util import find_spec
            >>> import torch
            >>> from flagquantum.qec import DemError, DetectorErrorModel
            >>> if find_spec("pymatching") is not None:
            ...     model = DetectorErrorModel(1, 1, (DemError(0.125, (0,), (0,)),))
            ...     decoder = PyMatchingDecoder.from_detector_error_model(model)
            ...     bits, weights = decoder.decode_batch_with_weights(
            ...         torch.tensor([[0], [1]], dtype=torch.uint8)
            ...     )
            ...     assert bits.tolist() == [[False], [True]]
            ...     assert weights.shape == (2,) and weights[0] == 0
        """
        if (
            type(bit_packed_shots) is not bool
            or type(bit_packed_predictions) is not bool
        ):
            raise TypeError("packed flags must be bool")
        if bit_packed_shots:
            if not isinstance(detection_events, torch.Tensor):
                raise TypeError("detection_events must be a torch.Tensor")
            if (
                detection_events.layout != torch.strided
                or detection_events.dtype != torch.uint8
            ):
                raise TypeError("packed detection_events must be a dense uint8 tensor")
            if detection_events.device.type != "cpu":
                raise CapabilityError("batch matching accepts CPU tensors only")
            width = (self.graph.num_detectors + 7) // 8
            if detection_events.ndim != 2 or detection_events.shape[1] != width:
                raise ValueError(
                    f"packed detection_events must have shape (shots, {width})"
                )
            remainder = self.graph.num_detectors % 8
            if remainder and bool((detection_events[:, -1] >> remainder).any()):
                raise ValueError("unused packed detector padding bits must be zero")
        else:
            _check_batch_input(detection_events, self.graph.num_detectors)
        shots = detection_events.shape[0]
        columns = (
            (self.graph.num_observables + 7) // 8
            if bit_packed_predictions
            else self.graph.num_observables
        )
        dtype = torch.uint8 if bit_packed_predictions else torch.bool
        if shots == 0:
            return torch.empty((0, columns), dtype=dtype), torch.empty(
                0, dtype=torch.float64
            )
        syndrome = detection_events.to(dtype=torch.uint8, copy=True).contiguous()
        try:
            raw = self._matching.decode_batch(
                syndrome.numpy(),
                return_weights=True,
                bit_packed_shots=bit_packed_shots,
                bit_packed_predictions=bit_packed_predictions,
            )
        except ValueError as exc:
            raise CapabilityError(
                f"PyMatching could not decode the weighted batch: {exc}"
            ) from exc
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise CapabilityError("PyMatching returned an invalid weighted result pair")
        try:
            predictions, weights = (torch.as_tensor(item) for item in raw)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CapabilityError(
                "PyMatching returned an invalid weighted buffer"
            ) from exc
        if (
            predictions.device.type != "cpu"
            or predictions.layout != torch.strided
            or predictions.dtype != torch.uint8
            or predictions.shape != (shots, columns)
        ):
            raise CapabilityError("PyMatching returned invalid weighted predictions")
        if bit_packed_predictions:
            remainder = self.graph.num_observables % 8
            if remainder and bool((predictions[:, -1] >> remainder).any()):
                raise CapabilityError("PyMatching returned nonzero logical padding")
        elif not bool(((predictions == 0) | (predictions == 1)).all()):
            raise CapabilityError("PyMatching returned nonbinary predictions")
        if (
            weights.device.type != "cpu"
            or weights.layout != torch.strided
            or weights.dtype != torch.float64
            or weights.shape != (shots,)
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise CapabilityError("PyMatching returned invalid SDK matching weights")
        return predictions.to(dtype=dtype).clone(), weights.clone()

    def decode(self, detection_events: Iterable[int]) -> MatchingDecodeResult:
        """Return PyMatching's cheapest explanation of a syndrome.

        Args:
            detection_events: Indices of the detectors that flipped, in any
                order.

        Returns:
            The logical observables PyMatching predicts, the mechanisms it
            selected, and their exact total weight.

        Raises:
            CapabilityError: PyMatching found no perfect matching, which is the
                syndrome this package's matcher refuses as well, or it selected
                an edge this graph does not carry.
            ValueError: A detector index is outside the graph, or the syndrome
                names one detector twice.
        """

        defects = _defects(detection_events, num_detectors=self.graph.num_detectors)
        syndrome = [0] * self.graph.num_detectors
        for detector in defects:
            syndrome[detector] = 1
        # One SDK solution carries both logical labels and private edge labels.
        # Keep the stored logical-only graph, its batch path and instance fields
        # unchanged; the additional graph is local to this full-record call.
        matching, _ = _translate(self.graph, record_edge_ids=True)
        try:
            predicted = matching.decode(syndrome)
        except ValueError as exc:
            raise CapabilityError(
                "PyMatching found no perfect matching for a syndrome the "
                f"self-implemented matcher refuses as well: {exc}"
            ) from exc
        edges = tuple(
            edge
            for index, edge in enumerate(self.graph.edges)
            if predicted[self.graph.num_observables + index]
        )
        return MatchingDecodeResult(
            observables=tuple(
                index
                for index, flipped in enumerate(predicted[: self.graph.num_observables])
                if flipped
            ),
            error_edges=edges,
            weight=sum(edge.weight for edge in edges),
        )


__all__ = (
    "MatchingDependencyError",
    "PyMatchingDecoder",
)


_retain_module(
    "flagquantum.qec.adapters",
    MatchingDependencyError,
    _pymatching,
    _translate,
    PyMatchingDecoder,
)
