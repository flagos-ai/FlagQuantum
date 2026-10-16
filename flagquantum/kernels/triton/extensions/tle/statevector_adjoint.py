"""FlagTree TLE implementation of sharded statevector adjoint work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

if TYPE_CHECKING:
    from ..._jit import jit
else:
    from triton import jit


@jit(do_not_specialize=["rank_basis"])
def _complex64_sharded_1q_vjp_adjoint_tle_kernel(
    local_before_parts: tl.tensor,
    remote_before_parts: tl.tensor,
    local_adjoint_parts: tl.tensor,
    remote_adjoint_parts: tl.tensor,
    matrix_parts: tl.tensor,
    derivative_parts: tl.tensor,
    next_adjoint_parts: tl.tensor,
    partial_gradients: tl.tensor,
    element_count: tl.tensor,
    rank_basis: tl.tensor,
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    valid = offset < element_count

    local_before_real = tle.load(
        local_before_parts + 2 * offset,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    local_before_imag = tle.load(
        local_before_parts + 2 * offset + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    remote_before_real = tle.load(
        remote_before_parts + 2 * offset,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    remote_before_imag = tle.load(
        remote_before_parts + 2 * offset + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    local_adjoint_real = tle.load(
        local_adjoint_parts + 2 * offset,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    local_adjoint_imag = tle.load(
        local_adjoint_parts + 2 * offset + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    remote_adjoint_real = tle.load(
        remote_adjoint_parts + 2 * offset,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    remote_adjoint_imag = tle.load(
        remote_adjoint_parts + 2 * offset + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )

    before0_real = tl.where(rank_basis == 0, local_before_real, remote_before_real)
    before0_imag = tl.where(rank_basis == 0, local_before_imag, remote_before_imag)
    before1_real = tl.where(rank_basis == 0, remote_before_real, local_before_real)
    before1_imag = tl.where(rank_basis == 0, remote_before_imag, local_before_imag)
    adjoint0_real = tl.where(rank_basis == 0, local_adjoint_real, remote_adjoint_real)
    adjoint0_imag = tl.where(rank_basis == 0, local_adjoint_imag, remote_adjoint_imag)
    adjoint1_real = tl.where(rank_basis == 0, remote_adjoint_real, local_adjoint_real)
    adjoint1_imag = tl.where(rank_basis == 0, remote_adjoint_imag, local_adjoint_imag)

    # FlagTree 0.7.0 cannot lower scalar async loads. State and adjoint vectors
    # use TLE while the four selected matrix and derivative scalars remain
    # synchronous, matching the established provider capability boundary.
    row = rank_basis * 4
    d0_real = tl.load(derivative_parts + row)
    d0_imag = tl.load(derivative_parts + row + 1)
    d1_real = tl.load(derivative_parts + row + 2)
    d1_imag = tl.load(derivative_parts + row + 3)
    derivative_real = (
        d0_real * before0_real
        - d0_imag * before0_imag
        + d1_real * before1_real
        - d1_imag * before1_imag
    )
    derivative_imag = (
        d0_real * before0_imag
        + d0_imag * before0_real
        + d1_real * before1_imag
        + d1_imag * before1_real
    )
    contribution = (
        local_adjoint_real * derivative_real + local_adjoint_imag * derivative_imag
    )
    tl.store(partial_gradients + program, tl.sum(contribution, axis=0))

    column = rank_basis * 2
    m0_real = tl.load(matrix_parts + column)
    m0_imag = tl.load(matrix_parts + column + 1)
    m1_real = tl.load(matrix_parts + 4 + column)
    m1_imag = tl.load(matrix_parts + 5 + column)
    next_real = (
        m0_real * adjoint0_real
        + m0_imag * adjoint0_imag
        + m1_real * adjoint1_real
        + m1_imag * adjoint1_imag
    )
    next_imag = (
        m0_real * adjoint0_imag
        - m0_imag * adjoint0_real
        + m1_real * adjoint1_imag
        - m1_imag * adjoint1_real
    )
    tl.store(next_adjoint_parts + 2 * offset, next_real, mask=valid)
    tl.store(next_adjoint_parts + 2 * offset + 1, next_imag, mask=valid)


def launch_complex64_sharded_1q_vjp_adjoint_tle(
    local_before: torch.Tensor,
    remote_before: torch.Tensor,
    local_adjoint: torch.Tensor,
    remote_adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative_matrix: torch.Tensor,
    *,
    rank_basis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the authorized TLE sharded one-qubit VJP implementation."""

    next_adjoint = torch.empty_like(local_adjoint)
    element_count = local_before.numel()
    block = 256
    program_count = triton.cdiv(element_count, block)
    partial_gradients = torch.empty(
        program_count,
        dtype=torch.float32,
        device=local_before.device,
    )
    _complex64_sharded_1q_vjp_adjoint_tle_kernel[(program_count,)](
        torch.view_as_real(local_before),
        torch.view_as_real(remote_before),
        torch.view_as_real(local_adjoint),
        torch.view_as_real(remote_adjoint),
        torch.view_as_real(matrix),
        torch.view_as_real(derivative_matrix),
        torch.view_as_real(next_adjoint),
        partial_gradients,
        element_count,
        int(rank_basis),
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return next_adjoint, partial_gradients.sum()


__all__ = ("launch_complex64_sharded_1q_vjp_adjoint_tle",)
