"""Vendor-neutral weighted matching data, distinct from probability-owned DEMs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import torch

from flagquantum.errors import CapabilityError


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

    @classmethod
    def from_check_matrix(
        cls,
        check_matrix: torch.Tensor,
        *,
        weights: float | torch.Tensor = 1.0,
        faults_matrix: torch.Tensor | None = None,
        error_probabilities: float | torch.Tensor | None = None,
        repetitions: int = 1,
        timelike_weights: float | torch.Tensor = 1.0,
        measurement_error_probabilities: float | torch.Tensor | None = None,
        merge_strategy: Literal[
            "disallow", "independent", "smallest-weight", "keep-original", "replace"
        ] = "disallow",
        use_virtual_boundary_node: bool = True,
    ) -> tuple[MatchingGraph, tuple[int | None, ...]]:
        """Import binary check/fault incidence with explicit costs and merge policy.

        Returns the graph and first-round source-column to final-edge mapping.
        Spatial rounds are ordered first; time-like edges are detector-major.
        Ignored zero-support columns map to None. Costs and rates are independent.

        Examples:
            >>> import torch
            >>> graph, columns = MatchingGraph.from_check_matrix(torch.eye(2, dtype=torch.uint8))
            >>> columns
            (0, 1)
        """
        checks = _binary_matrix(check_matrix, "check_matrix")
        rows, columns = checks.shape
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, Integral)
            or repetitions < 1
        ):
            raise ValueError("repetitions must be a positive integer")
        if type(use_virtual_boundary_node) is not bool:
            raise TypeError("use_virtual_boundary_node must be bool")
        if merge_strategy not in (
            "disallow",
            "independent",
            "smallest-weight",
            "keep-original",
            "replace",
        ):
            raise ValueError("unknown merge strategy")
        faults = (
            torch.eye(columns, dtype=torch.int64)
            if faults_matrix is None
            else _binary_matrix(faults_matrix, "faults_matrix")
        )
        if faults.shape[1] != columns:
            raise ValueError("faults_matrix must have the check matrix column count")
        costs = _vector(weights, columns, "weights")
        rates = _rates(error_probabilities, columns, "error_probabilities")
        temporal_costs = _vector(timelike_weights, rows, "timelike_weights")
        temporal_rates = _rates(
            measurement_error_probabilities, rows, "measurement_error_probabilities"
        )
        edges: list[MatchingEdge] = []
        positions: dict[tuple[int, ...], int] = {}
        first_band: list[int | None] = [None] * columns
        supports = [
            tuple(int(i) for i in checks[:, column].nonzero().flatten())
            for column in range(columns)
        ]
        if any(len(support) > 2 for support in supports):
            raise CapabilityError("a check matrix column must be graphlike")
        labels = [
            tuple(int(i) for i in faults[:, column].nonzero().flatten())
            for column in range(columns)
        ]

        def add(edge: MatchingEdge) -> int:
            existing = positions.get(edge.detectors)
            if existing is None:
                positions[edge.detectors] = len(edges)
                edges.append(edge)
                return len(edges) - 1
            old = edges[existing]
            if merge_strategy == "disallow":
                raise ValueError("parallel columns require an explicit merge strategy")
            if merge_strategy == "replace" or (
                merge_strategy == "smallest-weight" and edge.weight < old.weight
            ):
                edges[existing] = edge
            elif merge_strategy == "independent":
                weight = _logaddexp(0.0, old.weight + edge.weight) - _logaddexp(
                    old.weight, edge.weight
                )
                probability = (
                    None
                    if old.error_probability is None or edge.error_probability is None
                    else old.error_probability * (1 - edge.error_probability)
                    + edge.error_probability * (1 - old.error_probability)
                )
                edges[existing] = MatchingEdge(
                    old.detectors, weight, old.fault_ids, probability
                )
            return existing

        for round_index in range(int(repetitions)):
            for column, support in enumerate(supports):
                if support:
                    index = add(
                        MatchingEdge(
                            tuple(node + round_index * rows for node in support),
                            costs[column],
                            labels[column],
                            rates[column],
                        )
                    )
                    if round_index == 0:
                        first_band[column] = index
        for detector in range(rows):
            for round_index in range(int(repetitions) - 1):
                add(
                    MatchingEdge(
                        (
                            round_index * rows + detector,
                            (round_index + 1) * rows + detector,
                        ),
                        temporal_costs[detector],
                        (),
                        temporal_rates[detector],
                    )
                )
        nodes = rows * int(repetitions)
        boundaries: tuple[int, ...] = ()
        if not use_virtual_boundary_node and any(
            len(edge.detectors) == 1 for edge in edges
        ):
            boundaries = (nodes,)
            edges = [
                MatchingEdge(
                    (
                        (edge.detectors[0], nodes)
                        if len(edge.detectors) == 1
                        else edge.detectors
                    ),
                    edge.weight,
                    edge.fault_ids,
                    edge.error_probability,
                )
                for edge in edges
            ]
            nodes += 1
        return cls(nodes, faults.shape[0], tuple(edges), boundaries), tuple(first_band)

    def to_check_matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Export owned node/fault incidence and costs in this graph's edge order.

        Explicit boundary metadata and optional rates remain on the graph.

        Examples:
            >>> graph = MatchingGraph(1, 0, (MatchingEdge((0,), 2.0),))
            >>> graph.to_check_matrices()[2].tolist()
            [2.0]
        """
        checks = torch.zeros((self.num_nodes, len(self.edges)), dtype=torch.uint8)
        faults = torch.zeros((self.num_fault_ids, len(self.edges)), dtype=torch.uint8)
        for column, edge in enumerate(self.edges):
            for node in edge.detectors:
                checks[node, column] = 1
            for index in edge.fault_ids:
                faults[index, column] = 1
        return (
            checks,
            faults,
            torch.tensor([edge.weight for edge in self.edges], dtype=torch.float64),
        )


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


def _binary_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.ndim != 2
    ):
        raise TypeError(f"{name} must be a two-dimensional CPU tensor")
    if value.dtype not in (
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
        raise TypeError(f"{name} must have binary bool or integer dtype")
    if value.layout == torch.sparse_coo:
        if value.sparse_dim() != 2:
            raise TypeError(f"{name} must use two sparse coordinate dimensions")
        if not value.is_coalesced():
            raise ValueError(f"{name} COO coordinates must be coalesced")
        coordinates = value.indices()
        if any(
            bool((coordinates[axis] < 0).any())
            or bool((coordinates[axis] >= value.shape[axis]).any())
            for axis in range(2)
        ):
            raise ValueError(f"{name} has out-of-range COO coordinates")
    elif value.layout == torch.sparse_csr:
        crow, indices = value.crow_indices(), value.col_indices()
        if (
            crow.numel() != value.shape[0] + 1
            or indices.numel() != value.values().numel()
            or bool((crow < 0).any())
            or bool((crow > indices.numel()).any())
        ):
            raise ValueError(f"{name} has invalid CSR storage")
        if (
            int(crow[0]) != 0
            or int(crow[-1]) != indices.numel()
            or bool((crow[1:] < crow[:-1]).any())
        ):
            raise ValueError(f"{name} has invalid CSR row pointers")
        for row in range(value.shape[0]):
            cols = indices[int(crow[row]) : int(crow[row + 1])]
            if (
                bool((cols < 0).any())
                or bool((cols >= value.shape[1]).any())
                or bool((cols[1:] <= cols[:-1]).any())
            ):
                raise ValueError(f"{name} CSR columns must be canonical")
    elif value.layout != torch.strided:
        raise TypeError(f"{name} has unsupported sparse layout")
    result = value.to(dtype=torch.int64).clone()
    if result.layout != torch.strided:
        result = result.to_dense()
    if not bool(((result == 0) | (result == 1)).all()):
        raise ValueError(f"{name} must be binary")
    return result


def _vector(value: float | torch.Tensor, size: int, name: str) -> list[float]:
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.ndim != 1
            or value.shape[0] != size
            or value.dtype == torch.bool
            or value.is_complex()
        ):
            raise ValueError(
                f"{name} must be a real CPU vector with the declared width"
            )
        result = [_finite(float(item), name) for item in value]
    else:
        result = [_finite(value, name)] * size
    if any(abs(item) > 2**24 - 1 for item in result):
        raise ValueError(f"{name} exceeds the pinned SDK cost limit")
    return result


def _rates(
    value: float | torch.Tensor | None, size: int, name: str
) -> list[float | None]:
    if value is None:
        return [None] * size
    result = _vector(value, size, name)
    if any(not 0 <= rate <= 1 for rate in result):
        raise ValueError(f"{name} must be in [0, 1]")
    return list(result)


def _logaddexp(first: float, second: float) -> float:
    larger, smaller = max(first, second), min(first, second)
    return larger + math.log1p(math.exp(smaller - larger))
