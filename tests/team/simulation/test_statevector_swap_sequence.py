"""Tests for the CPU consecutive-SWAP statevector fast path."""

import pytest
import torch

from flagquantum import Circuit

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_swap_sequence_fusion_is_bitwise_exact_and_reduces_passes(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(3107)
    inputs = (
        torch.randn((2, 64), generator=generator)
        + 1j * torch.randn((2, 64), generator=generator)
    ).to(dtype)
    circuit = Circuit(6, bsz=2, dtype=dtype, inputs=inputs)
    circuit.swap(0, 5).swap(1, 4).swap(2, 3).swap(0, 2)

    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "0")
    expected = circuit.state(refresh=True)
    rollback = circuit._last_statevector_runtime

    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "1")
    actual = circuit.state(refresh=True)
    optimized = circuit._last_statevector_runtime

    assert torch.equal(actual, expected)
    assert optimized["permutation_gates"] == rollback["permutation_gates"] == 4
    assert optimized["statevector_apply_count"] == 1
    assert rollback["statevector_apply_count"] == 4


def test_cpu_swap_sequence_fusion_preserves_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.randn(64, dtype=torch.complex128, requires_grad=True)
    circuit = Circuit(6, dtype=torch.complex128, inputs=inputs)
    circuit.swap(0, 5).swap(1, 4).swap(2, 3)

    monkeypatch.setenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "1")
    loss = circuit.state(refresh=True).real.square().sum()
    (gradient,) = torch.autograd.grad(loss, inputs)

    assert gradient.shape == inputs.shape
    assert torch.isfinite(gradient).all()
