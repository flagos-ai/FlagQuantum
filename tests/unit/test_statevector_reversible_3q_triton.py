"""Correctness and refusal boundaries for the SV-013 prototype."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_reversible_3q import (
    apply_complex64_local_reversible_3q,
)
from flagquantum.simulation.matrices import FREDKIN_MATRIX, TOFFOLI_MATRIX

pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.triton]


def _reference(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int, int],
    operation: str,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    matrix = TOFFOLI_MATRIX if operation == "ccx" else FREDKIN_MATRIX
    return _apply_matrix_layout(
        state,
        matrix.to(device=state.device, dtype=state.dtype),
        qubits,
        state.shape[1].bit_length() - 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("operation", ("ccx", "cswap"))
@pytest.mark.parametrize("qubits", ((0, 1, 2), (0, 7, 11), (9, 2, 6), (11, 5, 0)))
def test_local_reversible_3q_matches_reference(
    operation: str,
    qubits: tuple[int, int, int],
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(261_013)
    state = torch.randn(
        3,
        1 << 12,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = apply_complex64_local_reversible_3q(
        state,
        qubits=qubits,
        operation=operation,
    )
    expected = _reference(state, qubits=qubits, operation=operation)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_reversible_3q_rejects_non_cuda_state() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_reversible_3q(
            torch.randn(1, 16, dtype=torch.complex64),
            qubits=(0, 1, 2),
            operation="ccx",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_reversible_3q_rejects_unsupported_contracts() -> None:
    state = torch.randn(2, 16, device="cuda", dtype=torch.complex64)

    with pytest.raises(ValueError, match="three distinct local qubits"):
        apply_complex64_local_reversible_3q(
            state,
            qubits=(0, 1, 1),
            operation="ccx",
        )
    with pytest.raises(ValueError, match="operation must"):
        apply_complex64_local_reversible_3q(
            state,
            qubits=(0, 1, 2),
            operation="swap",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="forward-only"):
        apply_complex64_local_reversible_3q(
            state.requires_grad_(),
            qubits=(0, 1, 2),
            operation="cswap",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_reversible_3q_supports_exact_aliasing() -> None:
    state = torch.randn(2, 1 << 10, device="cuda", dtype=torch.complex64)
    expected = _reference(state, qubits=(1, 8, 4), operation="cswap")

    actual = apply_complex64_local_reversible_3q(
        state,
        qubits=(1, 8, 4),
        operation="cswap",
        output=state,
    )

    assert actual.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
