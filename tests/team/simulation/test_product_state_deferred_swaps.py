"""Tests for lazy product-state SWAP materialization."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking.simulator_workload_corpus import build_workload
from flagquantum.simulation.statevector.wire_permutation import (
    _apply_wire_permutation_gather,
    _clear_wire_permutation_cache,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cached_wire_permutation_matches_axis_materialization_and_gradient(
    dtype: torch.dtype,
) -> None:
    source_wires = (4, 1, 3, 0, 2)
    actual_input = torch.randn(2, 2**5, dtype=dtype, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    axes = {wire: index + 1 for index, wire in enumerate(source_wires)}
    expected = (
        expected_input.reshape((2,) + (2,) * 5)
        .permute((0,) + tuple(axes[wire] for wire in sorted(source_wires)))
        .reshape(expected_input.shape)
        .contiguous()
    )

    _clear_wire_permutation_cache()
    actual = _apply_wire_permutation_gather(actual_input, source_wires)
    weights = torch.arange(actual.numel(), dtype=actual.real.dtype).reshape(
        actual.shape
    )
    actual_gradient = torch.autograd.grad((actual.real * weights).sum(), actual_input)[
        0
    ]
    expected_gradient = torch.autograd.grad(
        (expected.real * weights).sum(), expected_input
    )[0]

    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, expected_gradient)


def test_deferred_qft_swaps_match_eager_materialization_and_parameter_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager_angle = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    eager = build_workload("truncated_qft_statevector", n_qubits=18)
    eager.ry(0, eager_angle)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_DEFER_SWAP", "0")
    expected = eager.state(refresh=True)
    expected_gradient = torch.autograd.grad(expected.real.sum(), eager_angle)[0]

    deferred_angle = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    deferred = build_workload("truncated_qft_statevector", n_qubits=18)
    deferred.ry(0, deferred_angle)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_DEFER_SWAP", "1")
    actual = deferred.state(refresh=True)
    actual_gradient = torch.autograd.grad(actual.real.sum(), deferred_angle)[0]

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )
    assert eager._initial_state_workspace is None
    assert deferred._initial_state_workspace is None


def test_deferred_cross_component_swap_preserves_following_gates_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(angle: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(18, dtype=torch.complex128)
        return (
            circuit.h(0)
            .cx(0, 1)
            .h(2)
            .cx(2, 3)
            .swap(1, 2)
            .ry(1, angle)
            .cx(2, 4)
            .swap(0, 4)
            .h(4)
        )

    eager_angle = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_DEFER_SWAP", "0")
    expected = build(eager_angle).state(refresh=True)
    expected_gradient = torch.autograd.grad(expected.real.sum(), eager_angle)[0]

    deferred_angle = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_DEFER_SWAP", "1")
    actual = build(deferred_angle).state(refresh=True)
    actual_gradient = torch.autograd.grad(actual.real.sum(), deferred_angle)[0]

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )
