"""Native Triton application of local two-qubit Pauli rotations."""

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
def _complex64_local_pauli_rotation_kernel(
    state_parts: tl.tensor,
    rotation_parts: tl.tensor,
    output_parts: tl.tensor,
    amplitude_count: tl.tensor,
    work_count: tl.tensor,
    batch_count: tl.tensor,
    rotation_batch_stride: tl.tensor,
    first_bit_position: tl.tensor,
    second_bit_position: tl.tensor,
    PAULI_KIND: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
) -> None:
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = work_count * batch_count
    mask = linear < total
    batch = linear // work_count
    work = linear - batch * work_count

    rotation_index = batch * rotation_batch_stride
    c = tl.load(rotation_parts + 2 * rotation_index, mask=mask, other=0.0)
    s = tl.load(rotation_parts + 2 * rotation_index + 1, mask=mask, other=0.0)
    if PAULI_KIND == 2:
        amplitude = work
        parity = ((amplitude >> first_bit_position) & 1) ^ (
            (amplitude >> second_bit_position) & 1
        )
        phase = 1.0 - 2.0 * parity
        state_index = batch * amplitude_count + amplitude
        value_real = tl.load(state_parts + 2 * state_index, mask=mask, other=0.0)
        value_imag = tl.load(
            state_parts + 2 * state_index + 1,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output_parts + 2 * state_index,
            c * value_real + s * phase * value_imag,
            mask=mask,
        )
        tl.store(
            output_parts + 2 * state_index + 1,
            c * value_imag - s * phase * value_real,
            mask=mask,
        )
    else:
        # XX and YY partition the state into disjoint amplitude pairs. Insert
        # a zero at the first target bit, then obtain the partner by flipping
        # both target bits. Each pair is loaded once and produces both outputs.
        low_mask = (1 << first_bit_position) - 1
        low = work & low_mask
        amplitude = ((work - low) << 1) | low
        paired_amplitude = amplitude ^ (1 << first_bit_position)
        paired_amplitude = paired_amplitude ^ (1 << second_bit_position)
        parity = (amplitude >> second_bit_position) & 1
        phase = 1.0
        if PAULI_KIND == 1:
            phase = 2.0 * parity - 1.0

        state_index = batch * amplitude_count + amplitude
        paired_index = batch * amplitude_count + paired_amplitude
        value_real = tl.load(state_parts + 2 * state_index, mask=mask, other=0.0)
        value_imag = tl.load(
            state_parts + 2 * state_index + 1,
            mask=mask,
            other=0.0,
        )
        paired_real = tl.load(
            state_parts + 2 * paired_index,
            mask=mask,
            other=0.0,
        )
        paired_imag = tl.load(
            state_parts + 2 * paired_index + 1,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output_parts + 2 * state_index,
            c * value_real + s * phase * paired_imag,
            mask=mask,
        )
        tl.store(
            output_parts + 2 * state_index + 1,
            c * value_imag - s * phase * paired_real,
            mask=mask,
        )
        tl.store(
            output_parts + 2 * paired_index,
            c * paired_real + s * phase * value_imag,
            mask=mask,
        )
        tl.store(
            output_parts + 2 * paired_index + 1,
            c * paired_imag - s * phase * value_real,
            mask=mask,
        )


def apply_complex64_local_pauli_rotation_2q(
    state: torch.Tensor,
    rotation: torch.Tensor,
    *,
    qubits: tuple[int, int],
    pauli: str,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``exp(-i theta P/2)`` from ``cos(theta/2) + i sin(theta/2)``."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or state.is_conj()
        or state.is_neg()
    ):
        raise ValueError(
            "Triton Pauli rotation requires contiguous CUDA complex64 [B, 2**n]"
        )
    amplitude_count = int(state.shape[1])
    if (
        state.shape[0] < 1
        or amplitude_count < 4
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError("Triton Pauli rotation requires a nonempty power-of-two state")
    n_qubits = amplitude_count.bit_length() - 1
    normalized_qubits = tuple(int(qubit) for qubit in qubits)
    if (
        len(normalized_qubits) != 2
        or normalized_qubits[0] == normalized_qubits[1]
        or any(not 0 <= qubit < n_qubits for qubit in normalized_qubits)
    ):
        raise ValueError("Triton Pauli rotation requires two distinct local qubits")
    normalized_pauli = pauli.upper()
    if normalized_pauli not in {"XX", "YY", "ZZ"}:
        raise ValueError("Triton Pauli rotation supports XX, YY, or ZZ")
    if state.requires_grad or rotation.requires_grad:
        raise ValueError("Triton Pauli rotation is a forward-only kernel")
    if (
        rotation.device != state.device
        or rotation.dtype != torch.complex64
        or rotation.ndim != 1
        or rotation.numel() not in {1, state.shape[0]}
        or not rotation.is_contiguous()
        or rotation.is_conj()
        or rotation.is_neg()
    ):
        raise ValueError(
            "rotation must be contiguous CUDA complex64 with one or batch values"
        )
    rotation_stride = 0 if rotation.numel() == 1 else 1

    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
        or output.data_ptr() == state.data_ptr()
    ):
        raise ValueError(
            "Triton Pauli rotation output must be distinct, contiguous, and match input"
        )

    bit_positions = tuple(n_qubits - 1 - qubit for qubit in normalized_qubits)
    work_count = amplitude_count if normalized_pauli == "ZZ" else amplitude_count // 2
    block = 256
    _complex64_local_pauli_rotation_kernel[
        (triton.cdiv(state.shape[0] * work_count, block),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(rotation),
        torch.view_as_real(output),
        amplitude_count,
        work_count,
        state.shape[0],
        rotation_stride,
        bit_positions[0],
        bit_positions[1],
        PAULI_KIND={"XX": 0, "YY": 1, "ZZ": 2}[normalized_pauli],
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["apply_complex64_local_pauli_rotation_2q"]
