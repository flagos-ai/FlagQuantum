"""Tests for the native CPU CX/RZZ/SWAP segment fast path."""

import pytest
import torch

from flagquantum import Circuit
from flagquantum.simulation.native_cpu.permutation import (
    native_cpu_cx_rzz_swap_available,
)
from flagquantum.simulation.statevector.dense_fusion_cpu import (
    _apply_cx_rzz_swap_sequence,
)
from flagquantum.simulation.statevector.program import (
    _StatevectorCXSequenceRZZSwapStep,
)

pytestmark = pytest.mark.unit


def _segment_circuit(inputs: torch.Tensor, angle: float | torch.Tensor) -> Circuit:
    circuit = Circuit(16, dtype=inputs.dtype, inputs=inputs)
    for wire in range(15):
        circuit.cx(wire, wire + 1)
    circuit.rzz(0, 15, angle).swap(1, 14)
    return circuit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_cx_rzz_swap_fusion_matches_three_step_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    if not native_cpu_cx_rzz_swap_available():
        pytest.skip("native CPU extension is unavailable")
    generator = torch.Generator().manual_seed(4411)
    inputs = (
        torch.randn(1 << 16, generator=generator)
        + 1j * torch.randn(1 << 16, generator=generator)
    ).to(dtype)
    circuit = _segment_circuit(inputs, 0.173)

    monkeypatch.setenv("FQ_CPU_CX_RZZ_SWAP_FUSION", "0")
    expected = circuit.state(refresh=True)
    rollback = circuit._last_statevector_runtime

    monkeypatch.setenv("FQ_CPU_CX_RZZ_SWAP_FUSION", "1")
    actual = circuit.state(refresh=True)
    optimized = circuit._last_statevector_runtime

    tolerance = 2e-6 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert rollback["statevector_apply_count"] == 3
    assert optimized["statevector_apply_count"] == 1
    assert optimized["permutation_gates"] == 16
    assert optimized["diagonal_elementwise_gates"] == 1


def test_cpu_cx_rzz_swap_fusion_declines_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.randn(1 << 16, dtype=torch.complex128, requires_grad=True)
    angle = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)
    circuit = _segment_circuit(inputs, angle)

    monkeypatch.setenv("FQ_CPU_CX_RZZ_SWAP_FUSION", "1")
    loss = circuit.state(refresh=True).real.square().sum()
    input_gradient, angle_gradient = torch.autograd.grad(loss, (inputs, angle))

    assert circuit._last_statevector_runtime["statevector_apply_count"] == 3
    assert torch.isfinite(input_gradient).all()
    assert torch.isfinite(angle_gradient)


def test_cpu_cx_rzz_swap_fusion_reuses_supplied_state_scratch() -> None:
    if not native_cpu_cx_rzz_swap_available():
        pytest.skip("native CPU extension is unavailable")
    generator = torch.Generator().manual_seed(4412)
    state = torch.randn((1, 2**16), dtype=torch.complex128, generator=generator)
    step = _StatevectorCXSequenceRZZSwapStep(
        controls=tuple(range(15)),
        targets=tuple(range(1, 16)),
        rzz_wires=(0, 15),
        rzz_angle=0.173,
        swap_wires=(1, 14),
    )
    expected = _apply_cx_rzz_swap_sequence(step, state, 16)
    scratch = torch.empty_like(state)

    actual = _apply_cx_rzz_swap_sequence(step, state, 16, scratch=scratch)

    assert actual.data_ptr() == scratch.data_ptr()
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
