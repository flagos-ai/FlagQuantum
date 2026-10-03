"""FlagTree TLE implementation of local flat-statevector gates."""

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


@jit(do_not_specialize=["bit_position"])
def _complex64_local_1q_tle_kernel(
    state_parts: tl.tensor,
    matrix_parts: tl.tensor,
    output_parts: tl.tensor,
    pair_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    bit_position: tl.tensor,
    BLOCK_PAIRS: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    total_pairs = pair_count * batch_count
    valid = linear < total_pairs
    batch = linear // pair_count
    pair = linear - batch * pair_count
    low_mask = (1 << bit_position) - 1
    low = pair & low_mask
    base = ((pair - low) << 1) | low
    index0 = batch * state_batch_stride + base
    index1 = index0 + (1 << bit_position)

    a_real = tle.load(
        state_parts + 2 * index0,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    a_imag = tle.load(
        state_parts + 2 * index0 + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    b_real = tle.load(
        state_parts + 2 * index1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    b_imag = tle.load(
        state_parts + 2 * index1 + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )

    # FlagTree 0.7.0's async lowering accepts tensor loads but fails closed on
    # scalar loads. The eight matrix scalars deliberately retain ``tl.load``.
    m00_real = tl.load(matrix_parts)
    m00_imag = tl.load(matrix_parts + 1)
    m01_real = tl.load(matrix_parts + 2)
    m01_imag = tl.load(matrix_parts + 3)
    m10_real = tl.load(matrix_parts + 4)
    m10_imag = tl.load(matrix_parts + 5)
    m11_real = tl.load(matrix_parts + 6)
    m11_imag = tl.load(matrix_parts + 7)
    out0_real = (
        m00_real * a_real - m00_imag * a_imag + m01_real * b_real - m01_imag * b_imag
    )
    out0_imag = (
        m00_real * a_imag + m00_imag * a_real + m01_real * b_imag + m01_imag * b_real
    )
    out1_real = (
        m10_real * a_real - m10_imag * a_imag + m11_real * b_real - m11_imag * b_imag
    )
    out1_imag = (
        m10_real * a_imag + m10_imag * a_real + m11_real * b_imag + m11_imag * b_real
    )
    tl.store(output_parts + 2 * index0, out0_real, mask=valid)
    tl.store(output_parts + 2 * index0 + 1, out0_imag, mask=valid)
    tl.store(output_parts + 2 * index1, out1_real, mask=valid)
    tl.store(output_parts + 2 * index1 + 1, out1_imag, mask=valid)


def launch_complex64_local_1q_tle(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    bit_position: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Launch the already-authorized TLE local one-qubit implementation."""

    pair_count = state.shape[1] // 2
    block_pairs = 256
    grid = (triton.cdiv(pair_count * state.shape[0], block_pairs),)
    _complex64_local_1q_tle_kernel[grid](
        torch.view_as_real(state),
        torch.view_as_real(matrix),
        torch.view_as_real(output),
        pair_count,
        state.shape[0],
        state.stride(0),
        int(bit_position),
        BLOCK_PAIRS=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ("launch_complex64_local_1q_tle",)
