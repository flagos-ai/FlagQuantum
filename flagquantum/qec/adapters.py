"""Cross-check the self-implemented matcher against PyMatching.

:mod:`flagquantum.qec.matching` is the authority for a decoded syndrome, and its
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

from ..errors import CapabilityError
from .decoding_graph import DecodingGraph, DecodingGraphEdge
from .dem import DetectorErrorModel
from .matching import MatchingDecodeResult, _defects

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
) -> tuple[Any, dict[tuple[int, int], DecodingGraphEdge]]:
    """Build PyMatching's matching graph for one decoding graph.

    Returns the matching graph and the map from a node pair back to the decoding
    graph edge it stands for, which is how a selected edge is read as the
    mechanism this package selected.
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
    for edge in graph.edges:
        first, second = edge.detectors
        labels = set(edge.observables)
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
    matching.ensure_num_fault_ids(graph.num_observables)
    if matching.num_detectors != graph.num_detectors:
        raise CapabilityError(
            "PyMatching inferred "
            f"{matching.num_detectors} detectors from this graph's {graph.num_edges} "
            f"mechanisms, and the graph states {graph.num_detectors}; refusing "
            "rather than renumbering a syndrome"
        )
    return matching, carried


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

    def _pair(self, first: int, second: int) -> tuple[int, int]:
        """Read one PyMatching edge as the node pair this graph records."""

        if first == _BOUNDARY or second == _BOUNDARY:
            node = second if first == _BOUNDARY else first
            return (node, self.graph.boundary_node)
        return (first, second) if first < second else (second, first)

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
        try:
            predicted = self._matching.decode(syndrome)
            selected = self._matching.decode_to_edges_array(syndrome)
        except ValueError as exc:
            raise CapabilityError(
                "PyMatching found no perfect matching for a syndrome the "
                f"self-implemented matcher refuses as well: {exc}"
            ) from exc
        try:
            edges = tuple(
                self._edges[self._pair(int(row[0]), int(row[1]))] for row in selected
            )
        except KeyError as exc:  # pragma: no cover - a translation invariant
            raise CapabilityError(
                f"PyMatching selected the edge {exc.args[0]!r}, which this graph "
                "does not carry"
            ) from exc
        return MatchingDecodeResult(
            observables=tuple(
                index for index, flipped in enumerate(predicted) if flipped
            ),
            error_edges=edges,
            weight=sum(edge.weight for edge in edges),
        )


__all__ = (
    "MatchingDependencyError",
    "PyMatchingDecoder",
)
