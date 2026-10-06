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
    amplitude_count: tl.tensor,
    total_amplitudes: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < total_amplitudes
    amplitude = linear % amplitude_count
    first = (amplitude >> first_bit_position) & 1
    second = (amplitude >> second_bit_position) & 1
    source_amplitude = amplitude ^ ((first ^ second) << first_bit_position)
    source_amplitude = source_amplitude ^ ((first ^ second) << second_bit_position)
    batch_offset = linear - amplitude
    source = batch_offset + source_amplitude
    real = tl.load(state_parts + 2 * source, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * source + 1, mask=mask, other=0.0)
    tl.store(output_parts + 2 * linear, real, mask=mask)
    tl.store(output_parts + 2 * linear + 1, imag, mask=mask)


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
        or output.data_ptr() == state.data_ptr()
    ):
        raise ValueError(
            "Triton SWAP output must be distinct, contiguous, and match input"
        )

    bit_positions = tuple(n_qubits - 1 - qubit for qubit in normalized_qubits)
    total_amplitudes = state.numel()
    block = 256
    _complex64_local_swap_kernel[(triton.cdiv(total_amplitudes, block),)](
        torch.view_as_real(state),
        torch.view_as_real(output),
        amplitude_count,
        total_amplitudes,
        bit_positions[0],
        bit_positions[1],
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_swap"]
