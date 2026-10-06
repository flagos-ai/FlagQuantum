"""Correctness and refusal boundaries for the SV-012 prototype."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_controlled_matrix import (
    apply_complex64_local_controlled_1q,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.triton]


def _reference(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    control_qubit: int,
    target_qubit: int,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    batch = state.shape[0]
    matrices = matrix.reshape(-1, 2, 2)
    if matrices.shape[0] == 1:
        matrices = matrices.expand(batch, -1, -1)
    controlled = torch.zeros(batch, 4, 4, device=state.device, dtype=state.dtype)
    controlled[:, :2, :2] = torch.eye(2, device=state.device, dtype=state.dtype)
    controlled[:, 2:, 2:] = matrices
    return _apply_matrix_layout(
        state,
        controlled,
        (control_qubit, target_qubit),
        state.shape[1].bit_length() - 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("qubits", ((0, 1), (0, 11), (7, 2), (11, 0)))
@pytest.mark.parametrize("batched_matrix", (False, True))
def test_local_controlled_matrix_matches_reference(
    qubits: tuple[int, int],
    batched_matrix: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(261_012)
    state = torch.randn(
        3,
        1 << 12,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    shape = (3, 2, 2) if batched_matrix else (2, 2)
    matrix = torch.randn(
        *shape,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = apply_complex64_local_controlled_1q(
        state,
        matrix,
        control_qubit=qubits[0],
        target_qubit=qubits[1],
    )
    expected = _reference(
        state,
        matrix,
        control_qubit=qubits[0],
        target_qubit=qubits[1],
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_local_controlled_matrix_rejects_non_cuda_state() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_controlled_1q(
            torch.randn(1, 16, dtype=torch.complex64),
            torch.eye(2, dtype=torch.complex64),
            control_qubit=0,
            target_qubit=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_controlled_matrix_rejects_unsupported_contracts() -> None:
    state = torch.randn(2, 16, device="cuda", dtype=torch.complex64)
    matrix = torch.eye(2, device="cuda", dtype=torch.complex64)

    with pytest.raises(ValueError, match="distinct local control"):
        apply_complex64_local_controlled_1q(
            state,
            matrix,
            control_qubit=1,
            target_qubit=1,
        )
    with pytest.raises(ValueError, match=r"\[2, 2\] or \[B, 2, 2\]"):
        apply_complex64_local_controlled_1q(
            state,
            torch.eye(4, device="cuda", dtype=torch.complex64),
            control_qubit=0,
            target_qubit=1,
        )
    with pytest.raises(ValueError, match="forward-only"):
        apply_complex64_local_controlled_1q(
            state.requires_grad_(),
            matrix,
            control_qubit=0,
            target_qubit=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_controlled_matrix_supports_exact_aliasing() -> None:
    state = torch.randn(2, 1 << 10, device="cuda", dtype=torch.complex64)
    matrix = torch.randn(2, 2, device="cuda", dtype=torch.complex64)
    expected = _reference(
        state,
        matrix,
        control_qubit=1,
        target_qubit=8,
    )

    actual = apply_complex64_local_controlled_1q(
        state,
        matrix,
        control_qubit=1,
        target_qubit=8,
        output=state,
    )

    assert actual.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
