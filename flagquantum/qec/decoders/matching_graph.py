"""Vendor-neutral weighted matching data, distinct from probability-owned DEMs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


def _indices(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in values
    ):
        raise ValueError(f"{name} must contain nonnegative integer indices")
    result = tuple(sorted(int(value) for value in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not repeat an index")
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MatchingEdge:
    """An optimization edge with independent cost, fault labels and noise rate.

    Examples:
        >>> MatchingEdge((0,), -2.0, (1,), 0.1).weight
        -2.0
    """

    detectors: tuple[int, ...]
    weight: float
    fault_ids: tuple[int, ...] = ()
    error_probability: float | None = None

    def __post_init__(self) -> None:
        nodes = _indices(tuple(self.detectors), "detectors")
        if len(nodes) not in (1, 2):
            raise ValueError(
                "an edge requires one boundary endpoint or two distinct nodes"
            )
        weight = _finite(self.weight, "weight")
        if abs(weight) > 2**24 - 1:
            raise ValueError("weight exceeds the pinned SDK absolute limit")
        probability = self.error_probability
        if probability is not None:
            probability = _finite(probability, "error_probability")
            if not 0 <= probability <= 1:
                raise ValueError("error_probability must be in [0, 1]")
        object.__setattr__(self, "detectors", nodes)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self, "fault_ids", _indices(tuple(self.fault_ids), "fault_ids")
        )
        object.__setattr__(self, "error_probability", probability)


@dataclass(frozen=True)
class MatchingGraph:
    """An immutable weighted optimization graph with explicit shape and boundaries.

    Edge order is input order. Fault IDs do not identify edge positions.

    Examples:
        >>> graph = MatchingGraph(2, 1, (MatchingEdge((0, 1), 3.0, (0,)),))
        >>> graph.edges[0].weight
        3.0
    """

    num_nodes: int
    num_fault_ids: int
    edges: tuple[MatchingEdge, ...] = ()
    boundary_nodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("num_nodes", "num_fault_ids"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        boundaries = _indices(tuple(self.boundary_nodes), "boundary_nodes")
        if any(node >= self.num_nodes for node in boundaries):
            raise ValueError("boundary node outside graph")
        seen: set[tuple[int, ...]] = set()
        edges = tuple(self.edges)
        for edge in edges:
            if not isinstance(edge, MatchingEdge):
                raise TypeError("edges must contain MatchingEdge records")
            if any(node >= self.num_nodes for node in edge.detectors):
                raise ValueError("edge endpoint outside graph")
            if any(index >= self.num_fault_ids for index in edge.fault_ids):
                raise ValueError("edge fault ID outside graph shape")
            if edge.detectors in seen:
                raise ValueError(
                    "parallel endpoints require an explicit matrix merge policy"
                )
            seen.add(edge.detectors)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "boundary_nodes", boundaries)


@dataclass(frozen=True)
class MatchingSolution:
    """Selected fault/edge indices with distinct original and SDK costs.

    Examples:
        >>> MatchingSolution((0,), (1,), -2.0, -2.0).edge_indices
        (1,)
    """

    fault_ids: tuple[int, ...]
    edge_indices: tuple[int, ...]
    weight: float
    sdk_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fault_ids", _indices(tuple(self.fault_ids), "fault_ids")
        )
        object.__setattr__(
            self, "edge_indices", _indices(tuple(self.edge_indices), "edge_indices")
        )
        object.__setattr__(self, "weight", _finite(self.weight, "weight"))
        object.__setattr__(self, "sdk_weight", _finite(self.sdk_weight, "sdk_weight"))
