"""Native Triton application of a local statevector SWAP sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


@jit
def _swap_source_bit_positions(
    amplitude: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
) -> tl.tensor:
    first = (amplitude >> first_bit_position) & 1
    second = (amplitude >> second_bit_position) & 1
    difference = first ^ second
    amplitude = amplitude ^ (difference << first_bit_position)
    return amplitude ^ (difference << second_bit_position)


@jit(
    do_not_specialize=[
        "first_0",
        "second_0",
        "first_1",
        "second_1",
        "first_2",
        "second_2",
        "first_3",
        "second_3",
        "first_4",
        "second_4",
        "first_5",
        "second_5",
        "first_6",
        "second_6",
        "first_7",
        "second_7",
    ]
)
def _complex64_local_swap_sequence_kernel(
    state_parts: tl.tensor,
    output_parts: tl.tensor,
    amplitude_count: tl.tensor,
    total_amplitudes: tl.tensor,
    first_0: tl.tensor,
    second_0: tl.tensor,
    first_1: tl.tensor,
    second_1: tl.tensor,
    first_2: tl.tensor,
    second_2: tl.tensor,
    first_3: tl.tensor,
    second_3: tl.tensor,
    first_4: tl.tensor,
    second_4: tl.tensor,
    first_5: tl.tensor,
    second_5: tl.tensor,
    first_6: tl.tensor,
    second_6: tl.tensor,
    first_7: tl.tensor,
    second_7: tl.tensor,
    SWAP_COUNT: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < total_amplitudes
    amplitude = linear % amplitude_count
    source = amplitude
    if SWAP_COUNT > 0:
        source = _swap_source_bit_positions(source, first_0, second_0)
    if SWAP_COUNT > 1:
        source = _swap_source_bit_positions(source, first_1, second_1)
    if SWAP_COUNT > 2:
        source = _swap_source_bit_positions(source, first_2, second_2)
    if SWAP_COUNT > 3:
        source = _swap_source_bit_positions(source, first_3, second_3)
    if SWAP_COUNT > 4:
        source = _swap_source_bit_positions(source, first_4, second_4)
    if SWAP_COUNT > 5:
        source = _swap_source_bit_positions(source, first_5, second_5)
    if SWAP_COUNT > 6:
        source = _swap_source_bit_positions(source, first_6, second_6)
    if SWAP_COUNT > 7:
        source = _swap_source_bit_positions(source, first_7, second_7)
    source = linear - amplitude + source
    real = tl.load(state_parts + 2 * source, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * source + 1, mask=mask, other=0.0)
    tl.store(output_parts + 2 * linear, real, mask=mask)
    tl.store(output_parts + 2 * linear + 1, imag, mask=mask)


def apply_complex64_local_swap_sequence(
    state: torch.Tensor,
    *,
    swaps: tuple[tuple[int, int], ...],
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply two through eight ordered local SWAP gates in one state pass."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError(
            "Triton SWAP sequence requires contiguous CUDA complex64 [B, 2**n]"
        )
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 4
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton SWAP sequence requires a nonempty power-of-two state")
    n_qubits = amplitude_count.bit_length() - 1
    normalized_swaps = tuple((int(first), int(second)) for first, second in swaps)
    if not 2 <= len(normalized_swaps) <= 8:
        raise ValueError("Triton SWAP sequence requires two through eight gates")
    if any(
        first == second or not 0 <= first < n_qubits or not 0 <= second < n_qubits
        for first, second in normalized_swaps
    ):
        raise ValueError("Triton SWAP sequence requires distinct local qubits")
    if state.requires_grad:
        raise ValueError("Triton SWAP sequence is a forward-only kernel")

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
        or output.data_ptr() == state.data_ptr()
    ):
        raise ValueError(
            "Triton SWAP sequence output must be distinct, contiguous, and match input"
        )

    reversed_bit_positions = [
        (n_qubits - 1 - first, n_qubits - 1 - second)
        for first, second in reversed(normalized_swaps)
    ]
    reversed_bit_positions.extend([(0, 0)] * (8 - len(reversed_bit_positions)))
    flattened_positions = tuple(
        position for pair in reversed_bit_positions for position in pair
    )
    total_amplitudes = state.numel()
    block = 256
    _complex64_local_swap_sequence_kernel[(triton.cdiv(total_amplitudes, block),)](
        torch.view_as_real(state),
        torch.view_as_real(output),
        amplitude_count,
        total_amplitudes,
        *flattened_positions,
        SWAP_COUNT=len(normalized_swaps),
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_swap_sequence"]
