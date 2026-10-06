"""Correctness and refusal boundaries for the SV-012 prototype."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_swap import apply_complex64_local_swap

pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.triton]


def _reference(state: torch.Tensor, qubits: tuple[int, int]) -> torch.Tensor:
    first, second = (int(qubit) + 1 for qubit in qubits)
    n_qubits = state.shape[1].bit_length() - 1
    return (
        state.reshape((state.shape[0],) + (2,) * n_qubits)
        .transpose(first, second)
        .reshape(state.shape)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("qubits", ((0, 1), (0, 11), (7, 2), (11, 0)))
def test_local_swap_matches_layout_reference(qubits: tuple[int, int]) -> None:
    generator = torch.Generator(device="cuda").manual_seed(261_012)
    state = torch.randn(
        3,
        1 << 12,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = apply_complex64_local_swap(state, qubits=qubits)
    expected = _reference(state, qubits)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_local_swap_rejects_non_cuda_state() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_swap(
            torch.randn(1, 16, dtype=torch.complex64),
            qubits=(0, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_swap_rejects_unsupported_contracts() -> None:
    state = torch.randn(2, 16, device="cuda", dtype=torch.complex64)

    with pytest.raises(ValueError, match="two distinct"):
        apply_complex64_local_swap(state, qubits=(1, 1))
    with pytest.raises(ValueError, match="two distinct"):
        apply_complex64_local_swap(state, qubits=(0, 4))
    with pytest.raises(ValueError, match="forward-only"):
        apply_complex64_local_swap(state.requires_grad_(), qubits=(0, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_swap_supports_exact_aliasing() -> None:
    state = torch.randn(2, 1 << 10, device="cuda", dtype=torch.complex64)
    expected = _reference(state, (1, 8))

    actual = apply_complex64_local_swap(state, qubits=(1, 8), output=state)

    assert actual.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
