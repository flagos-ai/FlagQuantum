"""Decoding graphs derived from detector error models.

A detector error model states, for every independent physical error mechanism,
which detectors and which logical observables that mechanism flips. When every
mechanism flips one or two detectors, those mechanisms are the edges of a graph
whose vertices are the detectors, and inverting the model becomes a problem of
pairing up defective detectors along the cheapest edges. This module derives
that graph from a model; the matcher in
:mod:`flagquantum.qec.matching` consumes it and nothing else.

An edge is weighted by the negative log-likelihood ratio ``log((1 - p) / p)`` of
the mechanism it comes from, so the weight of a set of edges is the negative log
of the likelihood that exactly those mechanisms fired. A mechanism that flips one
detector is a boundary edge: it connects that detector to the graph's single
boundary node, which stands for an error at the edge of the patch whose
detection event leaves the code.

Derivation is exact and refuses rather than approximates. A mechanism that flips
three or more detectors is a hyperedge and has no edge here, so it is refused as
a capability boundary instead of being projected onto a pair of detectors. Two
mechanisms that share a detector pair with different observable labels stay two
edges, because they are two different explanations of the same detection events.
A mechanism of probability zero can never fire and contributes no edge; a
mechanism that fires with probability above one half is refused, because its
weight would be negative and a negative-weight cycle would make the cheapest
explanation unbounded. A model that states two mechanisms are alternatives is
refused as a whole: one weight per mechanism is the weight of a fault that fires
alone, and a matcher given a group of alternatives would weigh every member as
though the others could fire with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

from ..errors import CapabilityError
from .dem import DetectorErrorModel, _count, _normalized_indices, _probability

__all__ = ("DecodingGraph", "DecodingGraphEdge")

_GRAPH_DETECTORS = 2


@dataclass(frozen=True)
class DecodingGraphEdge:
    """One error mechanism as an edge, with the observables it flips.

    ``detectors`` holds the two graph nodes the mechanism flips, ascending. A
    mechanism that flips one detector is a boundary edge, and its second node is
    the boundary node of the graph it belongs to. ``observables`` holds the
    logical observables the mechanism flips, ascending: the labels a shortest
    path carries are the exclusive-or of its edges' labels, which is the parity
    the model's own ``observables_flips_matrix`` states for the same mechanisms.
    """

    detectors: tuple[int, int]
    probability: float
    observables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.detectors)
        if len(nodes) != _GRAPH_DETECTORS:
            raise ValueError("a decoding-graph edge connects exactly two nodes")
        if any(
            isinstance(node, bool) or not isinstance(node, Integral) for node in nodes
        ):
            raise TypeError("decoding-graph edge nodes must be integers")
        if nodes[0] < 0 or nodes[1] <= nodes[0]:
            raise ValueError(
                "decoding-graph edge nodes must be ascending and non-negative"
            )
        object.__setattr__(self, "detectors", (int(nodes[0]), int(nodes[1])))
        probability = _probability(self.probability, name="edge probability")
        if probability == 0.0:
            raise ValueError(
                "a decoding-graph edge must have a positive probability, because an "
                "edge no mechanism can select is not part of the graph"
            )
        if probability > 0.5:
            raise ValueError(
                "a decoding-graph edge must be no more likely than not, because its "
                "weight is the negative log-likelihood ratio log((1 - p) / p)"
            )
        object.__setattr__(self, "probability", probability)
        object.__setattr__(
            self,
            "observables",
            _normalized_indices(self.observables, name="edge observables"),
        )

    @property
    def weight(self) -> float:
        """The mechanism's negative log-likelihood ratio ``log((1 - p) / p)``."""

        return math.log((1.0 - self.probability) / self.probability)


@dataclass(frozen=True)
class DecodingGraph:
    """The detectors and mechanisms of a detector error model as a weighted graph.

    Every mechanism the model states contributes one edge, and the model stays
    the only source of the graph's shape: the detector and observable counts come
    from the model, not from the code the model describes. The boundary node is
    ``num_detectors``, which no detector can be, so a boundary edge needs no
    separate record. Edges are ordered by their nodes, then their observables,
    then their probability, so two graphs of the same model are equal.
    """

    num_detectors: int
    num_observables: int
    edges: tuple[DecodingGraphEdge, ...] = ()

    def __post_init__(self) -> None:
        detectors = _count(self.num_detectors, name="num_detectors")
        observables = _count(self.num_observables, name="num_observables")
        if detectors < 1:
            raise ValueError("a decoding graph requires at least one detector")
        object.__setattr__(self, "num_detectors", detectors)
        object.__setattr__(self, "num_observables", observables)
        for edge in self.edges:
            if not isinstance(edge, DecodingGraphEdge):
                raise TypeError("edges must contain DecodingGraphEdge records")
            first, second = edge.detectors
            if first >= detectors or second > detectors:
                raise ValueError(
                    f"edge {edge.detectors} is outside the graph's {detectors} "
                    "detectors and one boundary node"
                )
            for index in edge.observables:
                if index >= observables:
                    raise ValueError(
                        f"observable index {index} is outside the model shape"
                    )
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    self.edges,
                    key=lambda edge: (
                        edge.detectors,
                        edge.observables,
                        edge.probability,
                    ),
                )
            ),
        )

    @property
    def boundary_node(self) -> int:
        """The node a single-detector mechanism connects its detector to."""

        return self.num_detectors

    @property
    def num_edges(self) -> int:
        """Number of mechanisms the graph carries an edge for."""

        return len(self.edges)

    @classmethod
    def from_detector_error_model(cls, model: DetectorErrorModel) -> DecodingGraph:
        """Derive the graph a detector error model defines.

        Args:
            model: The model whose mechanisms become edges.

        Returns:
            The graph over the model's detectors, with the model's observable
            labels on its edges.

        Raises:
            CapabilityError: A mechanism flips three or more detectors, so the
                model is not graphlike and minimum-weight matching cannot
                represent it, or the model states that two mechanisms are
                alternatives, which one weight per edge cannot carry.
            ValueError: A mechanism fires with probability one or above one
                half, where no finite non-negative edge weight exists.
        """

        if not isinstance(model, DetectorErrorModel):
            raise TypeError("model must be a DetectorErrorModel")
        groups = model.exclusive_groups()
        if groups:
            shares = "; ".join(
                " and ".join(
                    "D" + ", D".join(str(index) for index in model.errors[at].detectors)
                    for at in indices
                )
                for indices in groups
            )
            raise CapabilityError(
                "minimum-weight matching weighs each mechanism by its own "
                "log-likelihood ratio, which assumes the mechanisms fire "
                "independently, and this model states that some of them are "
                f"alternatives: {shares}. A group of alternatives is one fault, "
                "whose weight is the negative log-likelihood of the group rather "
                "than of each member, so a matcher needs the group folded into a "
                "single mechanism before it can be given this model"
            )
        boundary = model.num_detectors
        edges: list[DecodingGraphEdge] = []
        for error in model.errors:
            if error.probability == 0.0:
                continue
            if len(error.detectors) > _GRAPH_DETECTORS:
                targets = ", ".join(f"D{index}" for index in error.detectors)
                raise CapabilityError(
                    "minimum-weight matching needs a graphlike detector error "
                    f"model, and the mechanism flipping {targets} is a hyperedge; "
                    "decompose the error into graphlike mechanisms or use a "
                    "hyperedge-aware decoder"
                )
            if not error.detectors:
                # An error with no detection event never changes the syndrome, so
                # no decoder that reads the syndrome can act on it.
                continue
            detector = error.detectors[0]
            second = error.detectors[1] if len(error.detectors) == 2 else boundary
            edges.append(
                DecodingGraphEdge(
                    detectors=(detector, second),
                    probability=error.probability,
                    observables=error.observables,
                )
            )
        return cls(
            num_detectors=model.num_detectors,
            num_observables=model.num_observables,
            edges=tuple(edges),
        )
