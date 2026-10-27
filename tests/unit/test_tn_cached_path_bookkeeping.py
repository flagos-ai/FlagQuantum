from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.tensor_network import path_search
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy

pytestmark = pytest.mark.unit


def _ring_network(width: int = 6, depth: int = 2) -> tuple[list, tuple]:
    """Build a two-layer ring of small tensors with a shared physical label."""

    generator = torch.Generator().manual_seed(7)
    nodes: list[TensorNetworkNode] = []
    for layer in range(depth):
        for bond in range(width):
            left = layer * width + bond
            right = layer * width + (bond + 1) % width
            physical = 1000 + bond
            tensor = torch.randn(2, 2, 2, generator=generator, dtype=torch.float64)
            nodes.append(TensorNetworkNode(tensor, (left, right, physical), name=left))
    return nodes, ()


def _einsum_reference(nodes: list, output_labels: tuple) -> torch.Tensor:
    """Contract the same nodes through a single torch.einsum call."""

    characters = {}
    for node in nodes:
        for label in node.labels:
            characters.setdefault(label, chr(ord("a") + len(characters)))
    inputs = ["".join(characters[label] for label in node.labels) for node in nodes]
    equation = (
        ",".join(inputs) + "->" + "".join(characters[label] for label in output_labels)
    )
    return torch.einsum(equation, *(node.tensor.to(torch.complex128) for node in nodes))


def _assert_matches_reference(measured: torch.Tensor, nodes: list) -> None:
    """Compare a contraction against the einsum oracle on the real parts.

    The network is real, so the oracle's imaginary parts are zero by
    construction; comparing float64 real parts keeps the comparison in one
    precision instead of letting a dtype cast decide it.
    """

    reference = _einsum_reference(nodes, ())
    assert torch.all(reference.imag == 0)
    torch.testing.assert_close(
        torch.real(measured).to(torch.float64),
        torch.real(reference).to(torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_cached_execution_matches_an_independent_einsum() -> None:
    nodes, output_labels = _ring_network()
    measured, _ = _contract_nodes_greedy(nodes, output_labels)
    _assert_matches_reference(measured, nodes)


def test_cached_execution_matches_the_cold_execution_it_reuses() -> None:
    nodes, output_labels = _ring_network()
    cold_value, cold_steps = _contract_nodes_greedy(nodes, output_labels)
    warm_value, warm_steps = _contract_nodes_greedy(nodes, output_labels)

    torch.testing.assert_close(warm_value, cold_value)
    _assert_matches_reference(warm_value, nodes)
    # The recorded provenance must survive path reuse unchanged.
    assert warm_steps == cold_steps


def test_reusing_a_cached_path_does_not_rescan_the_network_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached path is validated once, not re-derived at every step.

    ``_label_counts`` scans every active node, so calling it inside the
    contraction loop makes a cached contraction quadratic in the node count. A
    reused path already answers the question its caller asked; the cached
    execution must scan the network once per call rather than once per step.
    """

    nodes, output_labels = _ring_network()
    _contract_nodes_greedy(nodes, output_labels)

    counts_scan_sizes: list[int] = []
    original = path_search._label_counts

    def recording_count(
        active: object,
    ) -> dict[int, int]:
        counts_scan_sizes.append(len(active))
        return original(active)

    monkeypatch.setattr(path_search, "_label_counts", recording_count)
    _contract_nodes_greedy(nodes, output_labels)

    assert counts_scan_sizes == [len(nodes)], counts_scan_sizes


def test_cold_planning_scans_each_step_and_cached_execution_scans_once() -> None:
    nodes, output_labels = _ring_network()
    _contract_nodes_greedy(nodes, output_labels)

    calls = {"count": 0}
    original_counts = path_search._label_counts
    original_dims = path_search._label_dims

    def counting_counts(active: object) -> dict[int, int]:
        calls["count"] += 1
        return original_counts(active)

    def counting_dims(active: object) -> dict[int, int]:
        calls["count"] += 1
        return original_dims(active)

    path_search._label_counts = counting_counts
    path_search._label_dims = counting_dims
    try:
        warm_calls = 0
        _contract_nodes_greedy(nodes, output_labels)
        warm_calls = calls["count"]
    finally:
        path_search._label_counts = original_counts
        path_search._label_dims = original_dims

    assert warm_calls <= 2
