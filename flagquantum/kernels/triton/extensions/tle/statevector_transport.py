"""FlagTree TLE control-subspace transport kernels."""

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


@jit(do_not_specialize=["bit_position", "compressed_start"])
def _complex64_control_one_pack_tle_kernel(
    state_parts: tl.tensor,
    packed_parts: tl.tensor,
    chunk_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    bit_position: tl.tensor,
    compressed_start: tl.tensor,
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = chunk_count * batch_count
    valid = linear < total
    batch = linear // chunk_count
    compressed = compressed_start + linear - batch * chunk_count
    low_mask = (1 << bit_position) - 1
    low = compressed & low_mask
    state_index = (
        batch * state_batch_stride
        + ((compressed - low) << 1)
        + low
        + (1 << bit_position)
    )
    real = tle.load(
        state_parts + 2 * state_index,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    imag = tle.load(
        state_parts + 2 * state_index + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    tl.store(packed_parts + 2 * linear, real, mask=valid)
    tl.store(packed_parts + 2 * linear + 1, imag, mask=valid)


@jit(do_not_specialize=["bit_position", "compressed_start"])
def _complex64_control_one_unpack_tle_kernel(
    packed_parts: tl.tensor,
    output_parts: tl.tensor,
    chunk_count: tl.tensor,
    batch_count: tl.tensor,
    output_batch_stride: tl.tensor,
    bit_position: tl.tensor,
    compressed_start: tl.tensor,
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = chunk_count * batch_count
    valid = linear < total
    batch = linear // chunk_count
    compressed = compressed_start + linear - batch * chunk_count
    low_mask = (1 << bit_position) - 1
    low = compressed & low_mask
    output_index = (
        batch * output_batch_stride
        + ((compressed - low) << 1)
        + low
        + (1 << bit_position)
    )
    real = tle.load(
        packed_parts + 2 * linear,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    imag = tle.load(
        packed_parts + 2 * linear + 1,
        mask=valid,
        other=0.0,
        is_async=True,
    )
    tl.store(output_parts + 2 * output_index, real, mask=valid)
    tl.store(output_parts + 2 * output_index + 1, imag, mask=valid)


def launch_complex64_control_one_pack_tle(
    state: torch.Tensor,
    *,
    bit_position: int,
    compressed_start: int,
    compressed_end: int,
) -> torch.Tensor:
    """Launch the authorized TLE control-subspace gather."""

    chunk_count = compressed_end - compressed_start
    packed = torch.empty(
        (state.shape[0], chunk_count),
        dtype=state.dtype,
        device=state.device,
    )
    block = 256
    _complex64_control_one_pack_tle_kernel[
        (triton.cdiv(chunk_count * state.shape[0], block),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(packed),
        chunk_count,
        state.shape[0],
        state.stride(0),
        int(bit_position),
        int(compressed_start),
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return packed


def launch_complex64_control_one_unpack_tle(
    packed: torch.Tensor,
    output: torch.Tensor,
    *,
    bit_position: int,
    compressed_start: int,
) -> None:
    """Launch the authorized TLE control-subspace scatter."""

    chunk_count = packed.shape[1]
    block = 256
    _complex64_control_one_unpack_tle_kernel[
        (triton.cdiv(chunk_count * packed.shape[0], block),)
    ](
        torch.view_as_real(packed),
        torch.view_as_real(output),
        chunk_count,
        packed.shape[0],
        output.stride(0),
        int(bit_position),
        int(compressed_start),
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )


__all__ = (
    "launch_complex64_control_one_pack_tle",
    "launch_complex64_control_one_unpack_tle",
)
