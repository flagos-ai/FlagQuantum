"""Exact minimum-weight matching over a decoding graph.

A syndrome is the set of detectors that flipped. A decoder inverts a detector
error model by choosing the mechanisms that most plausibly produced that set, and
when the model is graphlike the cheapest such choice is a minimum-weight T-join:
a set of mechanisms whose odd-degree detectors are exactly the defective ones.
This module computes that set exactly, and reports the logical observables it
flips.

The computation has two exact steps. The first searches the graph from each
defective detector and yields the cheapest chain of mechanisms to every other
detector and to the boundary. The second chooses, among all the ways to pair the
defective detectors up and to route some of them to the boundary, the one of
least total weight. That choice is a minimum-weight perfect matching on the
metric closure of the defects, and the module enumerates it, so the decoder
accepts at most :data:`_DEFAULT_DEFECT_BUDGET` defective detectors by default and
refuses a larger syndrome as a capability boundary rather than returning a
pairing that only looks cheapest.

The boundary node is a sink, not a waypoint, and that is what makes a syndrome's
parity irrelevant. A mechanism at the edge of the patch flips one detector, so a
model can flip an odd number of detectors while a T-join exists for an even set
only; a defective detector is therefore allowed to pair with the boundary
instead of with another detector, and any number of detectors may do so, because
each boundary mechanism is an independent explanation. Chains between two
detectors never pass through the boundary, which costs nothing: going out to the
boundary and back from two detectors costs the sum of their boundary distances
either way, and the two edge sets differ only in edges they share, which cancel.

Chains may pass through detectors that are themselves defective, and the edges
two chains share cancel in the symmetric difference, so the reported mechanisms
are exactly the parity of the chains that were paired.

The observable prediction is the exclusive-or of the selected mechanisms'
labels, which is the parity the model's own ``observables_flips_matrix`` states
for those mechanisms. It is a decision about which of the cheapest explanations
the decoder adopts, not an estimate of the probability that the observable
flipped.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

from ..errors import CapabilityError
from .decoding_graph import DecodingGraph, DecodingGraphEdge
from .dem import DetectorErrorModel

__all__ = ("MatchingDecodeResult", "MinimumWeightMatchingDecoder")

# The matcher enumerates the ways to pair the defective detectors, so the default
# bounds a syndrome at twenty defects: that is 2**20 states, which a local memory
# experiment of a distance-three patch reaches only at noise well above its
# threshold. A caller that raises the budget pays for it in that enumeration.
_DEFAULT_DEFECT_BUDGET = 20

# Marks a defective detector that the matcher routes to the boundary rather than
# pairing with another one.
_BOUNDARY_PARTNER = -1


@dataclass(frozen=True)
class _SearchTree:
    """The cheapest chain from one defective detector, per detector."""

    distances: tuple[float, ...]
    parent_edges: tuple[int, ...]


@dataclass(frozen=True)
class MatchingDecodeResult:
    """The mechanisms one decode selected and the observables they flip.

    ``error_edges`` is the symmetric difference of the cheapest chains the
    matcher paired defective detectors along, so an edge two chains share is
    cancelled rather than counted twice. ``observables`` holds the logical
    observables the matcher predicts have flipped, ascending, and ``weight`` is
    the total weight of the selected mechanisms, which is at most the weight of
    the chains that produced them because cancelled edges are not counted.
    """

    observables: tuple[int, ...]
    error_edges: tuple[DecodingGraphEdge, ...]
    weight: float

    def __post_init__(self) -> None:
        observables = tuple(self.observables)
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in observables
        ):
            raise TypeError("decoded observables must be integer indices")
        if any(index < 0 for index in observables):
            raise ValueError("decoded observables must be non-negative")
        if tuple(sorted(set(observables))) != observables:
            raise ValueError("decoded observables must be unique and ascending")
        object.__setattr__(
            self, "observables", tuple(int(index) for index in observables)
        )
        if any(not isinstance(edge, DecodingGraphEdge) for edge in self.error_edges):
            raise TypeError("selected mechanisms must be DecodingGraphEdge records")
        if len(set(self.error_edges)) != len(self.error_edges):
            raise ValueError("a correction selects a mechanism at most once")
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise TypeError("decode weight must be a real number")
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("decode weight must be finite")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class MinimumWeightMatchingDecoder:
    """Decode a syndrome by pairing its detectors along the cheapest mechanisms.

    The decoder reads the graph a detector error model defines and nothing else,
    so it applies to any graphlike model rather than to one code. It reports the
    mechanisms it selected, which is the correction a caller can compare against
    the noise it injected, and the logical observables those mechanisms flip.

    Attributes:
        graph: The weighted graph the decoder searches.
        max_defects: Largest syndrome the exact pairing accepts.
    """

    graph: DecodingGraph
    max_defects: int = _DEFAULT_DEFECT_BUDGET

    def __post_init__(self) -> None:
        if not isinstance(self.graph, DecodingGraph):
            raise TypeError("graph must be a DecodingGraph")
        if isinstance(self.max_defects, bool) or not isinstance(
            self.max_defects, Integral
        ):
            raise TypeError("defect budget must be an integer")
        if self.max_defects < 1:
            raise ValueError("defect budget must be at least one")
        object.__setattr__(self, "max_defects", int(self.max_defects))

    @classmethod
    def from_detector_error_model(
        cls,
        model: DetectorErrorModel,
        *,
        max_defects: int = _DEFAULT_DEFECT_BUDGET,
    ) -> MinimumWeightMatchingDecoder:
        """Build the decoder for the graph a detector error model defines.

        Raises:
            CapabilityError: The model is not graphlike, so this decoder cannot
                represent it.
            ValueError: A mechanism's probability has no finite non-negative
                edge weight.
        """

        return cls(
            graph=DecodingGraph.from_detector_error_model(model),
            max_defects=max_defects,
        )

    def decode(self, detection_events: Iterable[int]) -> MatchingDecodeResult:
        """Return the cheapest explanation of a syndrome and its logical flips.

        Args:
            detection_events: Indices of the detectors that flipped, in any
                order.

        Returns:
            The predicted logical observables, the mechanisms the matcher
            selected, and their total weight.

        Raises:
            CapabilityError: The syndrome holds more defective detectors than
                ``max_defects``, or no chain of mechanisms explains it.
            ValueError: A detector index is outside the graph, or the syndrome
                names one detector twice.
        """

        defects = _defects(detection_events, num_detectors=self.graph.num_detectors)
        if not defects:
            return MatchingDecodeResult(observables=(), error_edges=(), weight=0.0)
        if len(defects) > self.max_defects:
            raise CapabilityError(
                f"minimum-weight matching decodes at most {self.max_defects} "
                f"detection events per syndrome and was given {len(defects)}; the "
                "exact pairing enumerates the ways to pair the defective detectors"
            )
        adjacency = _detector_adjacency(self.graph)
        trees = tuple(
            _search(adjacency, defect, self.graph.num_detectors) for defect in defects
        )
        pairing = _pairing(
            _pair_weights(trees, defects), _boundary_weights(self.graph, trees)
        )
        if pairing is None:
            raise CapabilityError(
                "no chain of mechanisms explains the syndrome: "
                + ", ".join(f"D{index}" for index in defects)
                + " cannot be paired through the graph's edges or its boundary"
            )
        selected: set[int] = set()
        for defect, partner in enumerate(pairing):
            if partner == _BOUNDARY_PARTNER:
                self._route_to_boundary(defects[defect], trees[defect], selected)
            elif defect < partner:
                self._walk(trees[defect], defects[partner], defects[defect], selected)
        error_edges = tuple(
            edge for index, edge in enumerate(self.graph.edges) if index in selected
        )
        flipped: set[int] = set()
        for edge in error_edges:
            for index in edge.observables:
                if index in flipped:
                    flipped.remove(index)
                else:
                    flipped.add(index)
        return MatchingDecodeResult(
            observables=tuple(sorted(flipped)),
            error_edges=error_edges,
            weight=sum(edge.weight for edge in error_edges),
        )

    def _walk(
        self,
        tree: _SearchTree,
        target: int,
        origin: int,
        selected: set[int],
    ) -> None:
        """Toggle the edges of one chain, from ``origin`` out to ``target``."""

        node = target
        while node != origin:
            edge_index = tree.parent_edges[node]
            if edge_index < 0:
                raise CapabilityError(
                    f"the chain from D{origin} to D{target} has a gap at D{node}"
                )
            _toggle(selected, edge_index)
            near, far = self.graph.edges[edge_index].detectors
            node = far if near == node else near

    def _route_to_boundary(
        self, defect: int, tree: _SearchTree, selected: set[int]
    ) -> None:
        """Toggle the cheapest chain from one detector to the boundary."""

        best_weight = math.inf
        best_edge = -1
        best_node = -1
        for edge_index, edge in enumerate(self.graph.edges):
            near, far = edge.detectors
            if far != self.graph.boundary_node:
                continue
            reachable = tree.distances[near]
            candidate = reachable + edge.weight
            if candidate < best_weight:
                best_weight = candidate
                best_edge = edge_index
                best_node = near
        if best_edge < 0 or not math.isfinite(best_weight):
            raise CapabilityError(
                f"detector D{defect} cannot reach the graph's boundary"
            )
        self._walk(tree, best_node, defect, selected)
        _toggle(selected, best_edge)


def _toggle(selected: set[int], edge_index: int) -> None:
    """Add an edge to a correction, or cancel it if it is already there.

    Combining two chains is a symmetric difference, so this is the operation that
    makes the result a parity. The cancellation cannot fire while the pairing is
    minimal: a repeated edge contributes twice to the pairing's weight, and
    dropping both copies leaves the parity of every detector unchanged and lowers
    the weight, which would contradict the pairing being the cheapest one. It is
    kept because it is what the operation means, so a caller that pairs chains
    differently still gets a correction rather than an edge counted twice.
    """

    if edge_index in selected:
        selected.remove(edge_index)
    else:
        selected.add(edge_index)


def _defects(detection_events: Iterable[int], *, num_detectors: int) -> tuple[int, ...]:
    """Normalize a syndrome's defective detectors into ascending order."""

    events: list[int] = []
    for event in detection_events:
        if isinstance(event, bool) or not isinstance(event, Integral):
            raise TypeError("detection events must be integer detector indices")
        index = int(event)
        if not 0 <= index < num_detectors:
            raise ValueError(
                f"detection event D{index} is outside the graph's {num_detectors} "
                "detectors"
            )
        events.append(index)
    defects = tuple(sorted(events))
    if len(set(defects)) != len(defects):
        raise ValueError("a syndrome cannot name the same detector twice")
    return defects


def _detector_adjacency(
    graph: DecodingGraph,
) -> tuple[tuple[tuple[float, int, int], ...], ...]:
    """Return each detector's ``(weight, neighbour, edge index)`` entries.

    Only mechanisms that flip two detectors are entries: the boundary is a sink,
    so a chain between two detectors never uses a boundary mechanism, and a chain
    that leaves the patch reaches the boundary as its last step.
    """

    entries: list[list[tuple[float, int, int]]] = [
        [] for _ in range(graph.num_detectors)
    ]
    for index, edge in enumerate(graph.edges):
        near, far = edge.detectors
        if far == graph.boundary_node:
            continue
        weight = edge.weight
        entries[near].append((weight, far, index))
        entries[far].append((weight, near, index))
    return tuple(tuple(sorted(node_entries)) for node_entries in entries)


def _search(
    adjacency: tuple[tuple[tuple[float, int, int], ...], ...],
    source: int,
    num_detectors: int,
) -> _SearchTree:
    """Search the detectors from ``source`` for its cheapest chain to each one.

    Weights are non-negative, so Dijkstra's search applies. Among chains of equal
    weight the search settles the smallest node and then the smallest edge, so
    the tree is a function of the graph and the source alone.
    """

    distances: list[float] = [math.inf] * num_detectors
    parent_edges: list[int] = [-1] * num_detectors
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for weight, neighbour, edge_index in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                parent_edges[neighbour] = edge_index
                heapq.heappush(queue, (candidate, neighbour))
    return _SearchTree(distances=tuple(distances), parent_edges=tuple(parent_edges))


def _pair_weights(
    trees: tuple[_SearchTree, ...], defects: tuple[int, ...]
) -> tuple[tuple[float, ...], ...]:
    """Return the cheapest chain weight between every two defective detectors."""

    return tuple(tuple(tree.distances[defect] for defect in defects) for tree in trees)


def _boundary_weights(
    graph: DecodingGraph, trees: tuple[_SearchTree, ...]
) -> tuple[float, ...]:
    """Return each defective detector's cheapest chain weight to the boundary."""

    boundary_edges = [
        (edge.detectors[0], edge.weight)
        for edge in graph.edges
        if edge.detectors[1] == graph.boundary_node
    ]
    return tuple(
        min(
            (
                tree.distances[detector] + weight
                for detector, weight in boundary_edges
                if math.isfinite(tree.distances[detector])
            ),
            default=math.inf,
        )
        for tree in trees
    )


def _pairing(
    weights: tuple[tuple[float, ...], ...],
    boundary: tuple[float, ...],
) -> tuple[int, ...] | None:
    """Pair every defective detector, returning its partner or the boundary.

    ``weights[first][second]`` is the cheapest chain connecting two defective
    detectors and ``boundary[first]`` is the cheapest chain from one to the
    graph's boundary; both are infinity when no chain exists. The states are the
    sets of defects still to place, and each state places its lowest-numbered
    defect either with a higher-numbered one or on the boundary, which enumerates
    every pairing exactly. Returns ``None`` when no pairing is possible.
    """

    size = len(weights)
    values: list[float] = [math.inf] * (1 << size)
    choices: list[int] = [_BOUNDARY_PARTNER] * (1 << size)
    values[0] = 0.0
    for mask in range(1, 1 << size):
        first = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << first)
        best = boundary[first] + values[rest]
        best_partner = _BOUNDARY_PARTNER
        remaining = rest
        while remaining:
            second = (remaining & -remaining).bit_length() - 1
            remaining ^= 1 << second
            candidate = weights[first][second] + values[rest ^ (1 << second)]
            if candidate < best:
                best = candidate
                best_partner = second
        values[mask] = best
        choices[mask] = best_partner
    if math.isinf(values[-1]):
        return None
    pairing = [_BOUNDARY_PARTNER] * size
    mask = (1 << size) - 1
    while mask:
        first = (mask & -mask).bit_length() - 1
        second = choices[mask]
        pairing[first] = second
        if second != _BOUNDARY_PARTNER:
            pairing[second] = first
            mask ^= (1 << first) | (1 << second)
        else:
            mask ^= 1 << first
    return tuple(pairing)
