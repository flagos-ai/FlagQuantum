"""Native Triton application of a local statevector SWAP gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


@jit(do_not_specialize=["first_bit_position", "second_bit_position"])
def _complex64_local_swap_kernel(
    state_parts: tl.tensor,
    output_parts: tl.tensor,
    group_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
    BLOCK_GROUPS: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    total_groups = group_count * batch_count
    mask = linear < total_groups
    batch = linear // group_count
    group = linear - batch * group_count

    low_bit = tl.minimum(first_bit_position, second_bit_position)
    high_bit = tl.maximum(first_bit_position, second_bit_position)
    low_mask = (1 << low_bit) - 1
    low = group & low_mask
    with_low_hole = ((group - low) << 1) | low
    high_mask = (1 << high_bit) - 1
    below_high = with_low_hole & high_mask
    base = ((with_low_hole - below_high) << 1) | below_high

    first_stride = 1 << first_bit_position
    second_stride = 1 << second_bit_position
    batch_base = batch * state_batch_stride + base
    index00 = batch_base
    index01 = batch_base + second_stride
    index10 = batch_base + first_stride
    index11 = batch_base + first_stride + second_stride

    value00_real = tl.load(state_parts + 2 * index00, mask=mask, other=0.0)
    value00_imag = tl.load(state_parts + 2 * index00 + 1, mask=mask, other=0.0)
    value01_real = tl.load(state_parts + 2 * index01, mask=mask, other=0.0)
    value01_imag = tl.load(state_parts + 2 * index01 + 1, mask=mask, other=0.0)
    value10_real = tl.load(state_parts + 2 * index10, mask=mask, other=0.0)
    value10_imag = tl.load(state_parts + 2 * index10 + 1, mask=mask, other=0.0)
    value11_real = tl.load(state_parts + 2 * index11, mask=mask, other=0.0)
    value11_imag = tl.load(state_parts + 2 * index11 + 1, mask=mask, other=0.0)

    tl.store(output_parts + 2 * index00, value00_real, mask=mask)
    tl.store(output_parts + 2 * index00 + 1, value00_imag, mask=mask)
    tl.store(output_parts + 2 * index01, value10_real, mask=mask)
    tl.store(output_parts + 2 * index01 + 1, value10_imag, mask=mask)
    tl.store(output_parts + 2 * index10, value01_real, mask=mask)
    tl.store(output_parts + 2 * index10 + 1, value01_imag, mask=mask)
    tl.store(output_parts + 2 * index11, value11_real, mask=mask)
    tl.store(output_parts + 2 * index11 + 1, value11_imag, mask=mask)


def apply_complex64_local_swap(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exchange two local statevector qubits without a layout permutation."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError("Triton SWAP requires contiguous CUDA complex64 [B, 2**n]")
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 4
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton SWAP requires a nonempty power-of-two state")
    n_qubits = amplitude_count.bit_length() - 1
    normalized_qubits = tuple(int(qubit) for qubit in qubits)
    if (
        len(normalized_qubits) != 2
        or normalized_qubits[0] == normalized_qubits[1]
        or any(not 0 <= qubit < n_qubits for qubit in normalized_qubits)
    ):
        raise ValueError("Triton SWAP requires two distinct local qubits")
    if state.requires_grad:
        raise ValueError("Triton SWAP is a forward-only kernel")

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
    ):
        raise ValueError("Triton SWAP output must be contiguous and match input")

    bit_positions = tuple(n_qubits - 1 - qubit for qubit in normalized_qubits)
    group_count = amplitude_count // 4
    block_groups = 256
    _complex64_local_swap_kernel[
        (triton.cdiv(state.shape[0] * group_count, block_groups),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(output),
        group_count,
        state.shape[0],
        state.stride(0),
        bit_positions[0],
        bit_positions[1],
        BLOCK_GROUPS=block_groups,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_swap"]
