"""Native Triton application of local statevector diagonal operators."""

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
def _complex64_local_diagonal_kernel(
    state_parts: tl.tensor,
    diagonal_parts: tl.tensor,
    output_parts: tl.tensor,
    amplitude_count: tl.tensor,
    batch_count: tl.tensor,
    state_batch_stride: tl.tensor,
    diagonal_batch_stride: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
    QUBIT_COUNT: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = amplitude_count * batch_count
    mask = linear < total
    batch = linear // amplitude_count
    amplitude = linear - batch * amplitude_count

    basis = (amplitude >> first_bit_position) & 1
    if QUBIT_COUNT == 2:
        basis = (basis << 1) | ((amplitude >> second_bit_position) & 1)
    diagonal_index = batch * diagonal_batch_stride + basis
    factor_real = tl.load(
        diagonal_parts + 2 * diagonal_index,
        mask=mask,
        other=0.0,
    )
    factor_imag = tl.load(
        diagonal_parts + 2 * diagonal_index + 1,
        mask=mask,
        other=0.0,
    )

    state_index = batch * state_batch_stride + amplitude
    value_real = tl.load(state_parts + 2 * state_index, mask=mask, other=0.0)
    value_imag = tl.load(state_parts + 2 * state_index + 1, mask=mask, other=0.0)
    output_real = factor_real * value_real - factor_imag * value_imag
    output_imag = factor_real * value_imag + factor_imag * value_real
    tl.store(output_parts + 2 * linear, output_real, mask=mask)
    tl.store(output_parts + 2 * linear + 1, output_imag, mask=mask)


def apply_complex64_local_diagonal(
    state: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    qubits: tuple[int, ...],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one constant or batch-resolved local diagonal operator.

    ``qubits`` defines the basis order of the diagonal entries. The first qubit
    contributes the most significant operator-basis bit.
    """

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError(
            "Triton local diagonal requires contiguous CUDA complex64 [B, 2**n]"
        )
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 2
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton local diagonal requires a nonempty power-of-two state")
    if len(qubits) not in {1, 2}:
        raise ValueError("Triton local diagonal requires one or two qubits")
    normalized_qubits = tuple(int(qubit) for qubit in qubits)
    n_qubits = amplitude_count.bit_length() - 1
    if len(set(normalized_qubits)) != len(normalized_qubits) or any(
        not 0 <= qubit < n_qubits for qubit in normalized_qubits
    ):
        raise ValueError("Triton local diagonal requires distinct local qubits")
    if state.requires_grad or diagonal.requires_grad:
        raise ValueError("Triton local diagonal is a forward-only kernel")

    operator_size = 1 << len(normalized_qubits)
    if diagonal.ndim == 1:
        if diagonal.shape != (operator_size,):
            raise ValueError("diagonal length must equal 2**len(qubits)")
        diagonal_batch_stride = 0
    elif diagonal.ndim == 2:
        if diagonal.shape[1] != operator_size or diagonal.shape[0] not in {
            1,
            state.shape[0],
        }:
            raise ValueError(
                "batched diagonal must have shape [1 or batch, 2**len(qubits)]"
            )
        diagonal_batch_stride = 0 if diagonal.shape[0] == 1 else operator_size
    else:
        raise ValueError("diagonal must have rank one or two")
    diagonal = diagonal.to(device=state.device, dtype=state.dtype).contiguous()

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Triton local diagonal output must be contiguous and match the input"
        )

    bit_positions = tuple(n_qubits - 1 - qubit for qubit in normalized_qubits)
    second_bit_position = bit_positions[1] if len(bit_positions) == 2 else 0
    block = 256
    _complex64_local_diagonal_kernel[(triton.cdiv(state.numel(), block),)](
        torch.view_as_real(state),
        torch.view_as_real(diagonal),
        torch.view_as_real(output),
        amplitude_count,
        state.shape[0],
        state.stride(0),
        diagonal_batch_stride,
        bit_positions[0],
        second_bit_position,
        QUBIT_COUNT=len(bit_positions),
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_diagonal"]
