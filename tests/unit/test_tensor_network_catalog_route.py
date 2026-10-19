"""Contracts for the catalog-derived device route in tensor-network execution."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.complex_bmm_dispatch import (
    _layout_complex_bmm_declared,
    _layout_complex_bmm_declared_for,
)
from flagquantum.simulation.tensor_network import local, path_search
from flagquantum.simulation.tensor_network.models import TensorNetworkNode

pytestmark = pytest.mark.unit


def _nodes(
    rows: int = 2, middle: int = 3, columns: int = 4
) -> tuple[TensorNetworkNode, ...]:
    generator = torch.Generator().manual_seed(1319)
    return (
        TensorNetworkNode(
            torch.randn(rows, middle, dtype=torch.complex64, generator=generator),
            (0, 1),
            name="left",
        ),
        TensorNetworkNode(
            torch.randn(middle, columns, dtype=torch.complex64, generator=generator),
            (1, 2),
            name="right",
        ),
    )


@pytest.mark.parametrize(
    ("device_type", "dtype", "declared"),
    (
        ("cpu", torch.complex64, False),
        ("cuda", torch.complex64, True),
        ("cuda", torch.complex128, False),
        ("cuda", torch.float32, False),
        ("flagos", torch.complex64, False),
        ("mps", torch.complex64, False),
        ("meta", torch.complex64, False),
    ),
)
def test_tensor_network_route_question_follows_the_catalog(
    device_type: str, dtype: torch.dtype, declared: bool
) -> None:
    assert (
        _layout_complex_bmm_declared_for(device_type=device_type, dtype=dtype)
        is declared
    )


def test_tensor_network_route_question_reads_the_operands() -> None:
    left, right = _nodes()
    on_meta = torch.zeros(2, 2, dtype=torch.complex64, device="meta")

    assert not _layout_complex_bmm_declared(left.tensor, right.tensor)
    assert not _layout_complex_bmm_declared(on_meta, on_meta)
    assert not _layout_complex_bmm_declared(
        left.tensor, left.tensor.to(torch.complex128)
    )


def test_staged_contraction_route_follows_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = _nodes()
    expected = nodes[0].tensor @ nodes[1].tensor

    eager, eager_steps = path_search._contract_nodes_greedy(nodes, (0, 2))
    assert eager_steps

    monkeypatch.setattr(
        path_search, "_layout_complex_bmm_declared_for", lambda **kwargs: True
    )
    staged, staged_steps = path_search._contract_nodes_greedy(nodes, (0, 2))

    # The staged route returns the contracted tensor without recording a step
    # list, so an empty step tuple is what proves it ran.
    assert staged_steps == ()
    assert torch.allclose(staged, expected)


def test_pair_contraction_route_follows_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pair takes the fused route only where the catalog declares one.

    The route question is only asked when no cached contraction path exists, so
    both halves use a node shape no earlier test contracted. A cached profile
    short-circuits the question entirely and the assertion below would hold for
    the wrong reason.
    """

    reference_calls: list[str] = []
    fused_calls: list[str] = []
    reference = path_search._einsum_pair_by_labels
    fused = path_search.complex_einsum_pair

    def record_reference(*args: object, **kwargs: object) -> torch.Tensor:
        reference_calls.append("reference")
        return reference(*args, **kwargs)

    def record_fused(*args: object, **kwargs: object) -> torch.Tensor:
        fused_calls.append("fused")
        return fused(*args, **kwargs)

    monkeypatch.setattr(path_search, "_einsum_pair_by_labels", record_reference)
    monkeypatch.setattr(path_search, "complex_einsum_pair", record_fused)

    uncached_nodes = _nodes(middle=5)
    expected = uncached_nodes[0].tensor @ uncached_nodes[1].tensor
    result, _ = path_search._contract_nodes_greedy(uncached_nodes, (0, 2))

    assert torch.allclose(result, expected)
    assert reference_calls == ["reference"]
    assert fused_calls == []

    monkeypatch.setattr(
        path_search, "_layout_complex_bmm_declared", lambda left, right: True
    )
    fused_nodes = _nodes(middle=7)
    fused_expected = fused_nodes[0].tensor @ fused_nodes[1].tensor
    fused_result, _ = path_search._contract_nodes_greedy(fused_nodes, (0, 2))

    assert torch.allclose(fused_result, fused_expected)
    # The declared route is the fused one, so the reference path must not have
    # been reached a second time.
    assert fused_calls == ["fused"]
    assert reference_calls == ["reference"]


def test_compiled_structure_reuse_follows_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit(2).ry(0, 0.2).cx(0, 1)

    first = fq.simulation.tensor_network.run_tensor_network(circuit)
    program = circuit._backend_programs[("tensor_network", 1)]

    rebuilt = fq.simulation.tensor_network.run_tensor_network(circuit)
    assert rebuilt.plan.path is not program.path

    monkeypatch.setattr(
        local, "_layout_complex_bmm_declared_for", lambda **kwargs: True
    )
    reused = fq.simulation.tensor_network.run_tensor_network(circuit)

    assert reused.plan.path is program.path
    assert reused.plan.nodes is not first.plan.nodes
    assert torch.allclose(reused.state(), circuit.state(), atol=1e-6)
