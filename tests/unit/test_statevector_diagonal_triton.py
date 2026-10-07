"""Correctness and capability tests for SV-010."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_diagonal import (
    apply_complex64_local_diagonal,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton]


def _reference(
    state: torch.Tensor,
    diagonal: torch.Tensor,
    qubits: tuple[int, ...],
) -> torch.Tensor:
    n_qubits = state.shape[1].bit_length() - 1
    amplitudes = torch.arange(state.shape[1], device=state.device)
    basis = torch.zeros_like(amplitudes)
    for qubit in qubits:
        basis = (basis << 1) | ((amplitudes >> (n_qubits - 1 - qubit)) & 1)
    factors = diagonal[basis] if diagonal.ndim == 1 else diagonal[:, basis]
    return state * factors


@pytest.mark.gpu
def test_local_diagonal_matches_reference_and_exact_alias() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    generator = torch.Generator(device="cuda").manual_seed(43)
    state = torch.randn(
        2,
        64,
        generator=generator,
        dtype=torch.complex64,
        device="cuda",
    )
    for qubits in ((0,), (5,), (0, 1), (1, 0), (0, 5), (5, 0), (1, 4)):
        operator_size = 1 << len(qubits)
        for diagonal in (
            torch.randn(
                operator_size,
                generator=generator,
                dtype=torch.complex64,
                device="cuda",
            ),
            torch.randn(
                2,
                operator_size,
                generator=generator,
                dtype=torch.complex64,
                device="cuda",
            ),
        ):
            expected = _reference(state, diagonal, qubits)
            actual = apply_complex64_local_diagonal(
                state,
                diagonal,
                qubits=qubits,
            )
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

            aliased = state.clone()
            returned = apply_complex64_local_diagonal(
                aliased,
                diagonal,
                qubits=qubits,
                output=aliased,
            )
            assert returned.data_ptr() == aliased.data_ptr()
            torch.testing.assert_close(aliased, expected, atol=2e-6, rtol=2e-6)


def test_local_diagonal_rejects_non_cuda_state() -> None:
    state = torch.randn(2, 8, dtype=torch.complex64)
    diagonal = torch.randn(2, dtype=torch.complex64)

    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_diagonal(state, diagonal, qubits=(0,))


@pytest.mark.gpu
def test_local_diagonal_rejects_unsupported_contracts() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    state = torch.randn(2, 64, dtype=torch.complex64, device="cuda")
    diagonal = torch.randn(4, dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="one or two qubits"):
        apply_complex64_local_diagonal(state, diagonal, qubits=(0, 1, 2))
    with pytest.raises(ValueError, match="distinct local qubits"):
        apply_complex64_local_diagonal(state, diagonal, qubits=(1, 1))
    with pytest.raises(ValueError, match="length must equal"):
        apply_complex64_local_diagonal(state, diagonal[:3], qubits=(0, 1))
    with pytest.raises(ValueError, match="forward-only"):
        apply_complex64_local_diagonal(
            state.requires_grad_(),
            diagonal,
            qubits=(0, 1),
        )
