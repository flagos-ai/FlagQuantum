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
    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "0")
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


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_controlled_phase_graph_can_update_an_owned_state_in_place(dtype) -> None:
    generator = torch.Generator().manual_seed(216)
    state = torch.randn((2, 16), dtype=dtype, generator=generator)
    factors, wires = _controlled_phase_graph_factors_cpu(
        ((1, 0, math.pi / 4), (3, 2, math.pi / 8)),
        device=state.device,
        dtype=dtype,
    )
    expected = _apply_controlled_phase_graph_cpu(state.clone(), factors, wires, 4)
    storage = state.untyped_storage().data_ptr()

    actual = _apply_controlled_phase_graph_cpu(state, factors, wires, 4, inplace=True)

    assert actual is state
    assert actual.untyped_storage().data_ptr() == storage
    torch.testing.assert_close(actual, expected)


def test_qft_graphs_only_update_executor_owned_states_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.statevector.local as local_statevector

    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "0")
    circuit = Circuit(5, bsz=2, dtype=torch.complex128)
    circuit.h(0)
    _append_portable_controlled_phase(circuit, 1, 0, math.pi / 4)
    _append_portable_controlled_phase(circuit, 2, 0, math.pi / 8)
    circuit.h(1)
    _append_portable_controlled_phase(circuit, 2, 1, math.pi / 4)
    _append_portable_controlled_phase(circuit, 3, 1, math.pi / 8)

    calls: list[bool] = []
    apply_graph = local_statevector._apply_controlled_phase_graph_cpu

    def record_inplace(*args, inplace=False, **kwargs):
        result = apply_graph(*args, inplace=inplace, **kwargs)
        calls.append(result is args[0])
        return result

    monkeypatch.setattr(
        local_statevector, "_apply_controlled_phase_graph_cpu", record_inplace
    )
    selected = circuit.state(refresh=True)
    assert calls == [True, True]

    calls.clear()
    monkeypatch.setenv("FQ_CPU_INPLACE_DIAGONAL_GRAPHS", "0")
    rollback = circuit.state(refresh=True)

    assert calls == [False, False]
    torch.testing.assert_close(selected, rollback)


def test_product_state_graph_cache_isolated_by_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", "1")
    cached_dtypes = set()
    for dtype in (torch.complex64, torch.complex128):
        circuit = build_workload("truncated_qft_statevector", n_qubits=16)
        circuit.dtype = dtype
        circuit.state(refresh=True)
        cached_dtypes.update(
            value.dtype for value in circuit._statevector_fused_matrices.values()
        )
    assert cached_dtypes == {torch.complex64, torch.complex128}


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_truncated_qft_graph_uses_product_state_and_matches_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "0")
    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "0")
    graph_circuit = build_workload("truncated_qft_statevector", n_qubits=18)
    graph_circuit.dtype = dtype
    rollback_circuit = build_workload("truncated_qft_statevector", n_qubits=18)
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


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_EXECUTION", "0")
    circuit = build_workload("truncated_qft_statevector", n_qubits=18)
    circuit.dtype = dtype
    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "0")

    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "0")
    expected = circuit.state(refresh=True)
    rollback_applies = circuit._last_statevector_runtime["statevector_apply_count"]

    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "1")
    actual = circuit.state(refresh=True)
    fused_applies = circuit._last_statevector_runtime["statevector_apply_count"]

    torch.testing.assert_close(actual, expected)
    assert rollback_applies == 44
    assert fused_applies == 28


def test_product_state_qft_uses_hadamard_phase_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "0")
    circuit = build_workload("truncated_qft_statevector", n_qubits=18)

    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "0")
    expected = circuit.state(refresh=True)
    rollback_applies = circuit._last_statevector_runtime["statevector_apply_count"]

    monkeypatch.setenv("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", "1")
    actual = circuit.state(refresh=True)
    fused_applies = circuit._last_statevector_runtime["statevector_apply_count"]

    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert rollback_applies == 44
    assert fused_applies == 28
