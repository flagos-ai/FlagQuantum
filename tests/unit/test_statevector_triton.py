"""CUDA numerical contracts for generic statevector Triton kernels."""

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_adjoint import (
    fused_complex64_local_1q_reversible_vjp,
    fused_complex64_local_1q_vjp_adjoint,
    fused_complex64_sharded_1q_vjp_adjoint,
)
from flagquantum.kernels.triton.statevector_gates import (
    apply_complex64_local_1q,
    apply_complex64_transpose_1q_inplace,
)
from flagquantum.runtime.executors.statevector.layout import _local_bit_view

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu,
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


@pytest.mark.parametrize("bit", [0, 3, 8])
def test_fused_reversible_vjp_matches_pytorch(bit):
    generator = torch.Generator(device="cuda").manual_seed(3701 + bit)
    before = torch.randn(
        2, 1 << 10, dtype=torch.complex64, device="cuda", generator=generator
    )
    adjoint = torch.randn_like(before)
    angle = torch.tensor(0.37, device="cuda")
    cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
    matrix = torch.stack(
        (
            torch.stack((cosine, -1j * sine)),
            torch.stack((-1j * sine, cosine)),
        )
    ).to(torch.complex64)
    derivative = torch.stack(
        (
            torch.stack((-0.5 * sine, -0.5j * cosine)),
            torch.stack((-0.5j * cosine, -0.5 * sine)),
        )
    ).to(torch.complex64)
    ket = _reference_apply(before, matrix, bit).contiguous()
    expected_adjoint = _reference_apply(adjoint, matrix.mH, bit)
    derivative_state = _reference_apply(before, derivative, bit)
    expected_gradient = torch.real(torch.sum(torch.conj(adjoint) * derivative_state))

    gradient = fused_complex64_local_1q_reversible_vjp(
        ket,
        adjoint,
        matrix,
        derivative,
        bit_position=bit,
    )

    torch.testing.assert_close(ket, before, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(adjoint, expected_adjoint, atol=3e-5, rtol=3e-5)
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


_FLAT_LOCAL_LIMIT_RESERVE_BYTES = 4 << 30


def _require_flat_local_limit_fit(amplitudes: int) -> None:
    """Skip unless one device can hold ``amplitudes`` complex64 amplitudes."""

    free, _ = torch.cuda.mem_get_info()
    if free < amplitudes * 8 + _FLAT_LOCAL_LIMIT_RESERVE_BYTES:
        pytest.skip("the flat-local address limit does not fit on this device")


@pytest.mark.slow
def test_local_cx_reaches_the_flat_local_address_limit():
    """The published limit is the largest state the kernel permutes correctly.

    ``apply_complex64_local_cx_inplace`` turns an amplitude index into one
    linear element offset. The limit is measured on hardware rather than
    derived: this test fills a state at the limit, checks a sampled
    permutation, and confirms the launch wrapper refuses the next size up
    instead of faulting the CUDA context.
    """

    from flagquantum.kernels.triton import FLAT_LOCAL_MAX_AMPLITUDES
    from flagquantum.kernels.triton.statevector_gates import (
        apply_complex64_local_cx_inplace,
    )

    amplitudes = FLAT_LOCAL_MAX_AMPLITUDES
    _require_flat_local_limit_fit(amplitudes)

    state = torch.empty(1, amplitudes, dtype=torch.complex64, device="cuda")
    chunk = 1 << 24
    flat = state.view(-1)
    for start in range(0, amplitudes, chunk):
        stop = min(start + chunk, amplitudes)
        values = torch.arange(start, stop, dtype=torch.float32, device="cuda")
        flat[start:stop].real.copy_(values)
        flat[start:stop].imag.copy_(values / 1e6)

    control_bit = amplitudes.bit_length() - 2
    torch.cuda.synchronize()

    sampled = torch.tensor(
        [amplitudes - 1, amplitudes - 3, (amplitudes >> 1) | 1, amplitudes >> 1],
        dtype=torch.long,
        device="cuda",
    )
    controlled = ((sampled >> control_bit) & 1) == 1
    sources = torch.where(controlled, sampled ^ 1, sampled)
    before = state[0, sources].clone()

    apply_complex64_local_cx_inplace(
        state, control_bit_position=control_bit, target_bit_position=0
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(state[0, sampled], before)
    del state, before
    torch.cuda.empty_cache()

    oversized = torch.zeros(1, dtype=torch.complex64).expand(1, amplitudes * 2)
    with pytest.raises(ValueError, match="at most"):
        apply_complex64_local_cx_inplace(
            oversized, control_bit_position=control_bit, target_bit_position=0
        )
