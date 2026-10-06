"""Native Triton application of a local controlled one-qubit matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


@jit(do_not_specialize=["control_bit_position", "target_bit_position"])
def _complex64_local_controlled_1q_kernel(
    state_parts: tl.tensor,
    matrix_parts: tl.tensor,
    output_parts: tl.tensor,
    group_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    matrix_batch_stride: tl.tensor,
    control_bit_position: tl.tensor,
    target_bit_position: tl.tensor,
    BLOCK_GROUPS: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    total_groups = group_count * batch_count
    mask = linear < total_groups
    batch = linear // group_count
    group = linear - batch * group_count

    low_bit = tl.minimum(control_bit_position, target_bit_position)
    high_bit = tl.maximum(control_bit_position, target_bit_position)
    low_mask = (1 << low_bit) - 1
    low = group & low_mask
    with_low_hole = ((group - low) << 1) | low
    high_mask = (1 << high_bit) - 1
    below_high = with_low_hole & high_mask
    base = ((with_low_hole - below_high) << 1) | below_high

    control_stride = 1 << control_bit_position
    target_stride = 1 << target_bit_position
    batch_base = batch * state_batch_stride + base
    index00 = batch_base
    index01 = batch_base + target_stride
    index10 = batch_base + control_stride
    index11 = batch_base + control_stride + target_stride

    value00_real = tl.load(state_parts + 2 * index00, mask=mask, other=0.0)
    value00_imag = tl.load(state_parts + 2 * index00 + 1, mask=mask, other=0.0)
    value01_real = tl.load(state_parts + 2 * index01, mask=mask, other=0.0)
    value01_imag = tl.load(state_parts + 2 * index01 + 1, mask=mask, other=0.0)
    value10_real = tl.load(state_parts + 2 * index10, mask=mask, other=0.0)
    value10_imag = tl.load(state_parts + 2 * index10 + 1, mask=mask, other=0.0)
    value11_real = tl.load(state_parts + 2 * index11, mask=mask, other=0.0)
    value11_imag = tl.load(state_parts + 2 * index11 + 1, mask=mask, other=0.0)

    matrix_offset = 2 * batch * matrix_batch_stride
    m00_real = tl.load(matrix_parts + matrix_offset)
    m00_imag = tl.load(matrix_parts + matrix_offset + 1)
    m01_real = tl.load(matrix_parts + matrix_offset + 2)
    m01_imag = tl.load(matrix_parts + matrix_offset + 3)
    m10_real = tl.load(matrix_parts + matrix_offset + 4)
    m10_imag = tl.load(matrix_parts + matrix_offset + 5)
    m11_real = tl.load(matrix_parts + matrix_offset + 6)
    m11_imag = tl.load(matrix_parts + matrix_offset + 7)

    out10_real = (
        m00_real * value10_real
        - m00_imag * value10_imag
        + m01_real * value11_real
        - m01_imag * value11_imag
    )
    out10_imag = (
        m00_real * value10_imag
        + m00_imag * value10_real
        + m01_real * value11_imag
        + m01_imag * value11_real
    )
    out11_real = (
        m10_real * value10_real
        - m10_imag * value10_imag
        + m11_real * value11_real
        - m11_imag * value11_imag
    )
    out11_imag = (
        m10_real * value10_imag
        + m10_imag * value10_real
        + m11_real * value11_imag
        + m11_imag * value11_real
    )

    tl.store(output_parts + 2 * index00, value00_real, mask=mask)
    tl.store(output_parts + 2 * index00 + 1, value00_imag, mask=mask)
    tl.store(output_parts + 2 * index01, value01_real, mask=mask)
    tl.store(output_parts + 2 * index01 + 1, value01_imag, mask=mask)
    tl.store(output_parts + 2 * index10, out10_real, mask=mask)
    tl.store(output_parts + 2 * index10 + 1, out10_imag, mask=mask)
    tl.store(output_parts + 2 * index11, out11_real, mask=mask)
    tl.store(output_parts + 2 * index11 + 1, out11_imag, mask=mask)


def apply_complex64_local_controlled_1q(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    control_qubit: int,
    target_qubit: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a two-by-two matrix only where the control qubit is one."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError(
            "Triton controlled 1q requires contiguous CUDA complex64 [B, 2**n]"
        )
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 4
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton controlled 1q requires a nonempty power-of-two state")
    n_qubits = amplitude_count.bit_length() - 1
    control_qubit = int(control_qubit)
    target_qubit = int(target_qubit)
    if (
        control_qubit == target_qubit
        or not 0 <= control_qubit < n_qubits
        or not 0 <= target_qubit < n_qubits
    ):
        raise ValueError(
            "Triton controlled 1q requires distinct local control and target qubits"
        )
    if state.requires_grad or matrix.requires_grad:
        raise ValueError("Triton controlled 1q is a forward-only kernel")
    if (
        matrix.device != state.device
        or matrix.dtype != torch.complex64
        or matrix.ndim not in {2, 3}
        or matrix.shape[-2:] != (2, 2)
        or (matrix.ndim == 3 and matrix.shape[0] not in {1, state.shape[0]})
        or not matrix.is_contiguous()
        or matrix.is_conj()
        or matrix.is_neg()
    ):
        raise ValueError("matrix must be contiguous CUDA complex64 [2, 2] or [B, 2, 2]")

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
    ):
        raise ValueError("controlled 1q output must be contiguous and match input")

    matrix_batch_stride = 0 if matrix.ndim == 2 or matrix.shape[0] == 1 else 4
    control_bit_position = n_qubits - 1 - control_qubit
    target_bit_position = n_qubits - 1 - target_qubit
    group_count = amplitude_count // 4
    block_groups = 128
    _complex64_local_controlled_1q_kernel[
        (triton.cdiv(state.shape[0] * group_count, block_groups),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(matrix),
        torch.view_as_real(output),
        group_count,
        state.shape[0],
        state.stride(0),
        matrix_batch_stride,
        control_bit_position,
        target_bit_position,
        BLOCK_GROUPS=block_groups,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_controlled_1q"]
