"""CUDA numerical contracts for generic statevector Triton kernels."""

import pytest
import torch

from flagquantum.runtime.backends.statevector.layout import _local_bit_view
from flagquantum.runtime.backends.statevector.triton import (
    apply_complex64_local_1q,
    apply_complex64_transpose_1q_inplace,
    fused_complex64_local_1q_vjp_adjoint,
    fused_complex64_sharded_1q_vjp_adjoint,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _reference_apply(
    state: torch.Tensor, matrix: torch.Tensor, bit: int
) -> torch.Tensor:
    batch, size = state.shape
    high = size >> (bit + 1)
    paired = state.reshape(batch, high, 2, 1 << bit)
    return torch.einsum("ij,bhjw->bhiw", matrix, paired).reshape_as(state)


@pytest.mark.parametrize("bit", [0, 2, 7])
def test_generic_local_1q_matches_pytorch(bit):
    generator = torch.Generator(device="cuda").manual_seed(1701 + bit)
    state = torch.randn(
        2, 1 << 10, dtype=torch.complex64, device="cuda", generator=generator
    )
    matrix = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )
    actual = apply_complex64_local_1q(state, matrix, bit_position=bit)
    expected = _reference_apply(state, matrix, bit)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(("bit", "exchanged"), [(0, 0), (3, 1), (8, 0)])
def test_fused_transpose_1q_matches_unpack_then_gate(bit, exchanged):
    generator = torch.Generator(device="cuda").manual_seed(2203 + bit + exchanged)
    state = torch.randn(
        2, 1 << 10, dtype=torch.complex64, device="cuda", generator=generator
    )
    received = torch.randn(
        2, 1 << 9, dtype=torch.complex64, device="cuda", generator=generator
    )
    matrix = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )
    unpacked = state.clone()
    _local_bit_view(unpacked, bit_position=bit, bit_value=exchanged).copy_(
        received.reshape_as(
            _local_bit_view(unpacked, bit_position=bit, bit_value=exchanged)
        )
    )
    expected = _reference_apply(unpacked, matrix, bit)
    actual = state.clone()
    apply_complex64_transpose_1q_inplace(
        actual,
        received,
        matrix,
        bit_position=bit,
        exchanged_bit_value=exchanged,
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("bit", [0, 3, 8])
def test_fused_vjp_and_adjoint_match_pytorch(bit):
    generator = torch.Generator(device="cuda").manual_seed(2903 + bit)
    before = torch.randn(
        2, 1 << 10, dtype=torch.complex64, device="cuda", generator=generator
    )
    adjoint = torch.randn_like(before)
    matrix = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )
    derivative = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )

    next_adjoint, gradient = fused_complex64_local_1q_vjp_adjoint(
        before, adjoint, matrix, derivative, bit_position=bit
    )
    expected_adjoint = _reference_apply(adjoint, matrix.mH, bit)
    derivative_state = _reference_apply(before, derivative, bit)
    expected_gradient = torch.real(torch.sum(torch.conj(adjoint) * derivative_state))

    torch.testing.assert_close(next_adjoint, expected_adjoint, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(gradient, expected_gradient, atol=3e-3, rtol=3e-5)


@pytest.mark.parametrize("rank_basis", [0, 1])
def test_fused_sharded_vjp_and_adjoint_matches_global_pair(rank_basis):
    generator = torch.Generator(device="cuda").manual_seed(4709 + rank_basis)
    before = torch.randn(
        2, 2, 4096, dtype=torch.complex64, device="cuda", generator=generator
    )
    adjoint = torch.randn_like(before)
    matrix = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )
    derivative = torch.randn(
        2, 2, dtype=torch.complex64, device="cuda", generator=generator
    )

    next_adjoint, gradient = fused_complex64_sharded_1q_vjp_adjoint(
        before[:, rank_basis].contiguous(),
        before[:, 1 - rank_basis].contiguous(),
        adjoint[:, rank_basis].contiguous(),
        adjoint[:, 1 - rank_basis].contiguous(),
        matrix,
        derivative,
        rank_basis=rank_basis,
    )
    expected_adjoint = torch.einsum("ij,bjk->bik", matrix.mH, adjoint)
    derivative_state = torch.einsum("ij,bjk->bik", derivative, before)
    expected_gradient = torch.real(
        torch.sum(torch.conj(adjoint[:, rank_basis]) * derivative_state[:, rank_basis])
    )

    torch.testing.assert_close(
        next_adjoint, expected_adjoint[:, rank_basis], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(gradient, expected_gradient, atol=3e-3, rtol=3e-5)
