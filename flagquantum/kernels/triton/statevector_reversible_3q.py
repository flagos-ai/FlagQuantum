"""Native Triton application of fixed local three-qubit permutations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import triton
import triton.language as tl

from flagquantum.kernels.triton import FLAT_LOCAL_MAX_AMPLITUDES

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


@jit(
    do_not_specialize=[
        "first_bit_position",
        "second_bit_position",
        "third_bit_position",
    ]
)
def _complex64_local_reversible_3q_kernel(
    state_parts: tl.tensor,
    output_parts: tl.tensor,
    group_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
    third_bit_position: tl.tensor,
    IS_CCX: tl.constexpr,  # noqa: N803
    BLOCK_GROUPS: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    total_groups = group_count * batch_count
    mask = linear < total_groups
    batch = linear // group_count
    group = linear - batch * group_count

    low_bit = tl.minimum(
        first_bit_position, tl.minimum(second_bit_position, third_bit_position)
    )
    high_bit = tl.maximum(
        first_bit_position, tl.maximum(second_bit_position, third_bit_position)
    )
    middle_bit = (
        first_bit_position
        + second_bit_position
        + third_bit_position
        - low_bit
        - high_bit
    )

    low_mask = (1 << low_bit) - 1
    low = group & low_mask
    with_low_hole = ((group - low) << 1) | low
    middle_mask = (1 << middle_bit) - 1
    below_middle = with_low_hole & middle_mask
    with_middle_hole = ((with_low_hole - below_middle) << 1) | below_middle
    high_mask = (1 << high_bit) - 1
    below_high = with_middle_hole & high_mask
    base = ((with_middle_hole - below_high) << 1) | below_high

    first_stride = 1 << first_bit_position
    second_stride = 1 << second_bit_position
    third_stride = 1 << third_bit_position
    batch_base = batch * state_batch_stride + base
    index0 = batch_base
    index1 = batch_base + third_stride
    index2 = batch_base + second_stride
    index3 = batch_base + second_stride + third_stride
    index4 = batch_base + first_stride
    index5 = batch_base + first_stride + third_stride
    index6 = batch_base + first_stride + second_stride
    index7 = batch_base + first_stride + second_stride + third_stride

    value0_real = tl.load(state_parts + 2 * index0, mask=mask, other=0.0)
    value0_imag = tl.load(state_parts + 2 * index0 + 1, mask=mask, other=0.0)
    value1_real = tl.load(state_parts + 2 * index1, mask=mask, other=0.0)
    value1_imag = tl.load(state_parts + 2 * index1 + 1, mask=mask, other=0.0)
    value2_real = tl.load(state_parts + 2 * index2, mask=mask, other=0.0)
    value2_imag = tl.load(state_parts + 2 * index2 + 1, mask=mask, other=0.0)
    value3_real = tl.load(state_parts + 2 * index3, mask=mask, other=0.0)
    value3_imag = tl.load(state_parts + 2 * index3 + 1, mask=mask, other=0.0)
    value4_real = tl.load(state_parts + 2 * index4, mask=mask, other=0.0)
    value4_imag = tl.load(state_parts + 2 * index4 + 1, mask=mask, other=0.0)
    value5_real = tl.load(state_parts + 2 * index5, mask=mask, other=0.0)
    value5_imag = tl.load(state_parts + 2 * index5 + 1, mask=mask, other=0.0)
    value6_real = tl.load(state_parts + 2 * index6, mask=mask, other=0.0)
    value6_imag = tl.load(state_parts + 2 * index6 + 1, mask=mask, other=0.0)
    value7_real = tl.load(state_parts + 2 * index7, mask=mask, other=0.0)
    value7_imag = tl.load(state_parts + 2 * index7 + 1, mask=mask, other=0.0)

    tl.store(output_parts + 2 * index0, value0_real, mask=mask)
    tl.store(output_parts + 2 * index0 + 1, value0_imag, mask=mask)
    tl.store(output_parts + 2 * index1, value1_real, mask=mask)
    tl.store(output_parts + 2 * index1 + 1, value1_imag, mask=mask)
    tl.store(output_parts + 2 * index2, value2_real, mask=mask)
    tl.store(output_parts + 2 * index2 + 1, value2_imag, mask=mask)
    tl.store(output_parts + 2 * index3, value3_real, mask=mask)
    tl.store(output_parts + 2 * index3 + 1, value3_imag, mask=mask)
    tl.store(output_parts + 2 * index4, value4_real, mask=mask)
    tl.store(output_parts + 2 * index4 + 1, value4_imag, mask=mask)
    if IS_CCX:
        tl.store(output_parts + 2 * index5, value5_real, mask=mask)
        tl.store(output_parts + 2 * index5 + 1, value5_imag, mask=mask)
        tl.store(output_parts + 2 * index6, value7_real, mask=mask)
        tl.store(output_parts + 2 * index6 + 1, value7_imag, mask=mask)
        tl.store(output_parts + 2 * index7, value6_real, mask=mask)
        tl.store(output_parts + 2 * index7 + 1, value6_imag, mask=mask)
    else:
        tl.store(output_parts + 2 * index5, value6_real, mask=mask)
        tl.store(output_parts + 2 * index5 + 1, value6_imag, mask=mask)
        tl.store(output_parts + 2 * index6, value5_real, mask=mask)
        tl.store(output_parts + 2 * index6 + 1, value5_imag, mask=mask)
        tl.store(output_parts + 2 * index7, value7_real, mask=mask)
        tl.store(output_parts + 2 * index7 + 1, value7_imag, mask=mask)


def apply_complex64_local_reversible_3q(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int, int],
    operation: Literal["ccx", "cswap"],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply CCX or controlled-SWAP without a dense eight-by-eight matrix."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError(
            "Triton reversible 3q requires contiguous CUDA complex64 [B, 2**n]"
        )
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 8
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton reversible 3q requires a nonempty power-of-two state")
    if state.numel() > FLAT_LOCAL_MAX_AMPLITUDES:
        raise ValueError(
            "flat local statevector kernels address at most "
            f"{FLAT_LOCAL_MAX_AMPLITUDES} amplitudes, got {state.numel()}"
        )
    n_qubits = amplitude_count.bit_length() - 1
    first_qubit, second_qubit, third_qubit = qubits
    qubits = (int(first_qubit), int(second_qubit), int(third_qubit))
    if len(set(qubits)) != 3 or any(qubit < 0 or qubit >= n_qubits for qubit in qubits):
        raise ValueError("Triton reversible 3q requires three distinct local qubits")
    if operation not in {"ccx", "cswap"}:
        raise ValueError("operation must be 'ccx' or 'cswap'")
    if state.requires_grad:
        raise ValueError("Triton reversible 3q is a forward-only kernel")

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
    ):
        raise ValueError("reversible 3q output must be contiguous and match input")

    bit_positions = tuple(n_qubits - 1 - qubit for qubit in qubits)
    group_count = amplitude_count // 8
    is_ccx = operation == "ccx"
    block_groups = 64 if is_ccx else 128
    num_warps = 4 if is_ccx else 8
    _complex64_local_reversible_3q_kernel[
        (triton.cdiv(state.shape[0] * group_count, block_groups),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(output),
        group_count,
        state.shape[0],
        state.stride(0),
        *bit_positions,
        IS_CCX=is_ccx,
        BLOCK_GROUPS=block_groups,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_reversible_3q"]
