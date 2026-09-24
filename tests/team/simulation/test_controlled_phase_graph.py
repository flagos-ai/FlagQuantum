"""Tests for exact CPU controlled-phase graph compilation and execution."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum import Circuit
from flagquantum.benchmarking.simulator_workload_corpus import build_workload
from flagquantum.simulation.statevector.controlled_phase import (
    _apply_controlled_phase_graph_cpu,
    _controlled_phase_graph_factors_cpu,
    _fuse_controlled_phase_graphs,
)
from flagquantum.simulation.statevector.program import (
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorControlledPhaseGraphStep,
)

pytestmark = pytest.mark.unit


def _append_portable_controlled_phase(
    circuit: Circuit,
    control: int,
    target: int,
    half_angle: float,
) -> None:
    circuit.rz(control, theta=half_angle)
    circuit.rz(target, theta=half_angle)
    circuit.cx(control, target)
    circuit.rz(target, theta=-half_angle)
    circuit.cx(control, target)


def test_consecutive_controlled_phases_compile_to_one_weighted_graph() -> None:
    phases = (
        _StatevectorControlledPhaseDecompositionStep(1, 0, math.pi / 4),
        _StatevectorControlledPhaseDecompositionStep(2, 0, math.pi / 8),
        _StatevectorControlledPhaseDecompositionStep(3, 0, math.pi / 16),
    )

    optimized = _fuse_controlled_phase_graphs(phases)

    assert optimized == [
        _StatevectorControlledPhaseGraphStep(
            (
                (1, 0, math.pi / 4),
                (2, 0, math.pi / 8),
                (3, 0, math.pi / 16),
            )
        )
    ]


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_controlled_phase_graph_matches_rollback_and_input_gradient(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(215)
    inputs = torch.randn((2, 16), dtype=dtype, generator=generator, requires_grad=True)
    circuit = Circuit(4, dtype=dtype, inputs=inputs)
    circuit.h(0)
    _append_portable_controlled_phase(circuit, 1, 0, math.pi / 4)
    _append_portable_controlled_phase(circuit, 2, 0, math.pi / 8)
    _append_portable_controlled_phase(circuit, 3, 0, math.pi / 16)

    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradient = torch.autograd.grad(
        expected.real.sum(), inputs, retain_graph=True
    )[0]
    rollback_statistics = dict(circuit._last_statevector_runtime)

    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradient = torch.autograd.grad(actual.real.sum(), inputs)[0]
    graph_statistics = circuit._last_statevector_runtime

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert rollback_statistics["statevector_apply_count"] == 4
    assert graph_statistics["statevector_apply_count"] == 2
    assert graph_statistics["diagonal_fused_regions"] == 1
    assert graph_statistics["fused_gate_count"] == 15


def test_controlled_phase_graph_factors_are_dtype_isolated() -> None:
    edges = ((1, 0, math.pi / 4), (2, 0, math.pi / 8))

    factors64, wires64 = _controlled_phase_graph_factors_cpu(
        edges, device=torch.device("cpu"), dtype=torch.complex64
    )
    factors128, wires128 = _controlled_phase_graph_factors_cpu(
        edges, device=torch.device("cpu"), dtype=torch.complex128
    )

    assert wires64 == wires128 == (0, 1, 2)
    assert factors64.dtype == torch.complex64
    assert factors128.dtype == torch.complex128
    torch.testing.assert_close(factors64.to(torch.complex128), factors128)


def test_controlled_phase_graph_kernel_rejects_an_invalid_wire() -> None:
    state = torch.zeros((1, 8), dtype=torch.complex128)
    factors = torch.ones(4, dtype=torch.complex128)

    with pytest.raises(ValueError, match="outside the statevector"):
        _apply_controlled_phase_graph_cpu(state, factors, (0, 3), 3)


def test_product_state_graph_cache_isolated_by_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = build_workload("truncated_qft_statevector", n_wires=16)
    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "1")

    circuit.dtype = torch.complex64
    circuit.state(refresh=True)
    circuit.dtype = torch.complex128
    circuit.state(refresh=True)

    cached_dtypes = {
        value.dtype for value in circuit._statevector_fused_matrices.values()
    }
    assert cached_dtypes == {torch.complex64, torch.complex128}


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_truncated_qft_graph_uses_product_state_and_matches_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    graph_circuit = build_workload("truncated_qft_statevector", n_wires=18)
    graph_circuit.dtype = dtype
    rollback_circuit = build_workload("truncated_qft_statevector", n_wires=18)
    rollback_circuit.dtype = dtype

    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "0")
    expected = rollback_circuit.state(refresh=True)
    rollback_applies = rollback_circuit._last_statevector_runtime[
        "statevector_apply_count"
    ]

    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "1")
    actual = graph_circuit.state(refresh=True)
    graph_applies = graph_circuit._last_statevector_runtime["statevector_apply_count"]

    torch.testing.assert_close(actual, expected)
    assert graph_circuit._initial_state_workspace is None
    assert rollback_applies == 75
    assert graph_applies == 44
