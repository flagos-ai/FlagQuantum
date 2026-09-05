"""Triton kernels for rank-local exact-statevector operations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["bit_position"])
def _complex64_local_1q_kernel(
    state_parts,
    matrix_parts,
    output_parts,
    pair_count,
    batch_count,
    state_batch_stride,
    bit_position,
    BLOCK_PAIRS: tl.constexpr,  # noqa: N803
):
    linear = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    total_pairs = pair_count * batch_count
    mask = linear < total_pairs
    batch = linear // pair_count
    pair = linear - batch * pair_count
    low_mask = (1 << bit_position) - 1
    low = pair & low_mask
    base = ((pair - low) << 1) | low
    index0 = batch * state_batch_stride + base
    index1 = index0 + (1 << bit_position)

    a_real = tl.load(state_parts + 2 * index0, mask=mask, other=0.0)
    a_imag = tl.load(state_parts + 2 * index0 + 1, mask=mask, other=0.0)
    b_real = tl.load(state_parts + 2 * index1, mask=mask, other=0.0)
    b_imag = tl.load(state_parts + 2 * index1 + 1, mask=mask, other=0.0)

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
    tl.store(output_parts + 2 * index0, out0_real, mask=mask)
    tl.store(output_parts + 2 * index0 + 1, out0_imag, mask=mask)
    tl.store(output_parts + 2 * index1, out1_real, mask=mask)
    tl.store(output_parts + 2 * index1 + 1, out1_imag, mask=mask)


def apply_complex64_local_1q(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    bit_position: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an arbitrary 2x2 matrix without materializing basis indices."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
    ):
        raise ValueError("Triton local 1q requires contiguous CUDA complex64 [B, N]")
    if matrix.shape != (2, 2):
        raise ValueError("Triton local 1q requires a 2x2 matrix")
    if not 0 <= bit_position < (state.shape[1].bit_length() - 1):
        raise ValueError("bit_position is outside the local state address")
    matrix = matrix.to(device=state.device, dtype=state.dtype).contiguous()
    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
    ):
        raise ValueError("Triton local 1q output must match the input state")
    # Exact aliasing is safe: each program loads both amplitudes in its
    # disjoint pair before writing either output.
    pair_count = state.shape[1] // 2
    block_pairs = 256
    grid = (triton.cdiv(pair_count * state.shape[0], block_pairs),)
    _complex64_local_1q_kernel[grid](
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


@triton.jit(do_not_specialize=["bit_position", "exchanged_bit_value"])
def _complex64_transpose_1q_kernel(
    state_parts,
    received_parts,
    matrix_parts,
    pair_count,
    batch_count,
    state_batch_stride,
    bit_position,
    exchanged_bit_value,
    BLOCK_PAIRS: tl.constexpr,  # noqa: N803
):
    linear = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    total = pair_count * batch_count
    mask = linear < total
    batch = linear // pair_count
    pair = linear - batch * pair_count
    low_mask = (1 << bit_position) - 1
    low = pair & low_mask
    index0 = batch * state_batch_stride + ((pair - low) << 1) + low
    index1 = index0 + (1 << bit_position)
    retained_index = tl.where(exchanged_bit_value == 0, index1, index0)
    retained_real = tl.load(state_parts + 2 * retained_index, mask=mask, other=0.0)
    retained_imag = tl.load(state_parts + 2 * retained_index + 1, mask=mask, other=0.0)
    remote_real = tl.load(received_parts + 2 * linear, mask=mask, other=0.0)
    remote_imag = tl.load(received_parts + 2 * linear + 1, mask=mask, other=0.0)
    a_real = tl.where(exchanged_bit_value == 0, remote_real, retained_real)
    a_imag = tl.where(exchanged_bit_value == 0, remote_imag, retained_imag)
    b_real = tl.where(exchanged_bit_value == 0, retained_real, remote_real)
    b_imag = tl.where(exchanged_bit_value == 0, retained_imag, remote_imag)

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
    tl.store(state_parts + 2 * index0, out0_real, mask=mask)
    tl.store(state_parts + 2 * index0 + 1, out0_imag, mask=mask)
    tl.store(state_parts + 2 * index1, out1_real, mask=mask)
    tl.store(state_parts + 2 * index1 + 1, out1_imag, mask=mask)


def apply_complex64_transpose_1q_inplace(
    state: torch.Tensor,
    received: torch.Tensor,
    matrix: torch.Tensor,
    *,
    bit_position: int,
    exchanged_bit_value: int,
) -> torch.Tensor:
    """Fuse a received half-shard transpose with its immediately following 1q gate."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or received.device != state.device
        or received.dtype != state.dtype
        or received.numel() * 2 != state.numel()
    ):
        raise ValueError("fused transpose 1q requires matching CUDA complex64 buffers")
    if matrix.shape != (2, 2) or exchanged_bit_value not in {0, 1}:
        raise ValueError("fused transpose 1q requires a 2x2 matrix and bit value")
    local_bits = state.shape[1].bit_length() - 1
    if not 0 <= bit_position < local_bits:
        raise ValueError("bit_position is outside the local state address")
    matrix = matrix.to(device=state.device, dtype=state.dtype).contiguous()
    pair_count = state.shape[1] // 2
    block_pairs = 256
    _complex64_transpose_1q_kernel[
        (triton.cdiv(pair_count * state.shape[0], block_pairs),)
    ](
        torch.view_as_real(state),
        torch.view_as_real(received.contiguous()),
        torch.view_as_real(matrix),
        pair_count,
        state.shape[0],
        state.stride(0),
        int(bit_position),
        int(exchanged_bit_value),
        BLOCK_PAIRS=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return state


@triton.jit(do_not_specialize=["bit_position"])
def _complex64_local_1q_vjp_adjoint_kernel(
    before_parts,
    adjoint_parts,
    matrix_parts,
    derivative_parts,
    next_adjoint_parts,
    partial_gradients,
    pair_count,
    batch_count,
    state_batch_stride,
    bit_position,
    REVERSE_KET: tl.constexpr,  # noqa: N803
    BLOCK_PAIRS: tl.constexpr,  # noqa: N803
):
    program = tl.program_id(0)
    linear = program * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    total_pairs = pair_count * batch_count
    mask = linear < total_pairs
    batch = linear // pair_count
    pair = linear - batch * pair_count
    low_mask = (1 << bit_position) - 1
    low = pair & low_mask
    base = ((pair - low) << 1) | low
    index0 = batch * state_batch_stride + base
    index1 = index0 + (1 << bit_position)

    a_real = tl.load(before_parts + 2 * index0, mask=mask, other=0.0)
    a_imag = tl.load(before_parts + 2 * index0 + 1, mask=mask, other=0.0)
    b_real = tl.load(before_parts + 2 * index1, mask=mask, other=0.0)
    b_imag = tl.load(before_parts + 2 * index1 + 1, mask=mask, other=0.0)
    l0_real = tl.load(adjoint_parts + 2 * index0, mask=mask, other=0.0)
    l0_imag = tl.load(adjoint_parts + 2 * index0 + 1, mask=mask, other=0.0)
    l1_real = tl.load(adjoint_parts + 2 * index1, mask=mask, other=0.0)
    l1_imag = tl.load(adjoint_parts + 2 * index1 + 1, mask=mask, other=0.0)

    m00_real = tl.load(matrix_parts)
    m00_imag = tl.load(matrix_parts + 1)
    m01_real = tl.load(matrix_parts + 2)
    m01_imag = tl.load(matrix_parts + 3)
    m10_real = tl.load(matrix_parts + 4)
    m10_imag = tl.load(matrix_parts + 5)
    m11_real = tl.load(matrix_parts + 6)
    m11_imag = tl.load(matrix_parts + 7)
    if REVERSE_KET:
        before0_real = (
            m00_real * a_real
            + m00_imag * a_imag
            + m10_real * b_real
            + m10_imag * b_imag
        )
        before0_imag = (
            m00_real * a_imag
            - m00_imag * a_real
            + m10_real * b_imag
            - m10_imag * b_real
        )
        before1_real = (
            m01_real * a_real
            + m01_imag * a_imag
            + m11_real * b_real
            + m11_imag * b_imag
        )
        before1_imag = (
            m01_real * a_imag
            - m01_imag * a_real
            + m11_real * b_imag
            - m11_imag * b_real
        )
        tl.store(before_parts + 2 * index0, before0_real, mask=mask)
        tl.store(before_parts + 2 * index0 + 1, before0_imag, mask=mask)
        tl.store(before_parts + 2 * index1, before1_real, mask=mask)
        tl.store(before_parts + 2 * index1 + 1, before1_imag, mask=mask)
        a_real, a_imag = before0_real, before0_imag
        b_real, b_imag = before1_real, before1_imag
    d00_real = tl.load(derivative_parts)
    d00_imag = tl.load(derivative_parts + 1)
    d01_real = tl.load(derivative_parts + 2)
    d01_imag = tl.load(derivative_parts + 3)
    d10_real = tl.load(derivative_parts + 4)
    d10_imag = tl.load(derivative_parts + 5)
    d11_real = tl.load(derivative_parts + 6)
    d11_imag = tl.load(derivative_parts + 7)

    d0_real = (
        d00_real * a_real - d00_imag * a_imag + d01_real * b_real - d01_imag * b_imag
    )
    d0_imag = (
        d00_real * a_imag + d00_imag * a_real + d01_real * b_imag + d01_imag * b_real
    )
    d1_real = (
        d10_real * a_real - d10_imag * a_imag + d11_real * b_real - d11_imag * b_imag
    )
    d1_imag = (
        d10_real * a_imag + d10_imag * a_real + d11_real * b_imag + d11_imag * b_real
    )
    contribution = (
        l0_real * d0_real + l0_imag * d0_imag + l1_real * d1_real + l1_imag * d1_imag
    )
    tl.store(partial_gradients + program, tl.sum(contribution, axis=0))

    next0_real = (
        m00_real * l0_real
        + m00_imag * l0_imag
        + m10_real * l1_real
        + m10_imag * l1_imag
    )
    next0_imag = (
        m00_real * l0_imag
        - m00_imag * l0_real
        + m10_real * l1_imag
        - m10_imag * l1_real
    )
    next1_real = (
        m01_real * l0_real
        + m01_imag * l0_imag
        + m11_real * l1_real
        + m11_imag * l1_imag
    )
    next1_imag = (
        m01_real * l0_imag
        - m01_imag * l0_real
        + m11_real * l1_imag
        - m11_imag * l1_real
    )
    tl.store(next_adjoint_parts + 2 * index0, next0_real, mask=mask)
    tl.store(next_adjoint_parts + 2 * index0 + 1, next0_imag, mask=mask)
    tl.store(next_adjoint_parts + 2 * index1, next1_real, mask=mask)
    tl.store(next_adjoint_parts + 2 * index1 + 1, next1_imag, mask=mask)


def fused_complex64_local_1q_vjp_adjoint(
    before: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative_matrix: torch.Tensor,
    *,
    bit_position: int,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a scalar gate-parameter VJP with the inverse adjoint gate."""

    if (
        before.device.type != "cuda"
        or before.dtype != torch.complex64
        or before.ndim != 2
        or not before.is_contiguous()
        or adjoint.shape != before.shape
        or adjoint.device != before.device
        or adjoint.dtype != before.dtype
        or not adjoint.is_contiguous()
    ):
        raise ValueError("fused Triton VJP requires contiguous CUDA complex64 states")
    if matrix.shape != (2, 2) or derivative_matrix.shape != (2, 2):
        raise ValueError("fused Triton VJP requires 2x2 matrices")
    matrix = matrix.to(device=before.device, dtype=before.dtype).contiguous()
    derivative_matrix = derivative_matrix.to(
        device=before.device, dtype=before.dtype
    ).contiguous()
    next_adjoint = torch.empty_like(adjoint) if output is None else output
    if (
        next_adjoint.shape != adjoint.shape
        or next_adjoint.dtype != adjoint.dtype
        or next_adjoint.device != adjoint.device
    ):
        raise ValueError("fused VJP output must match the adjoint state")
    # Exact adjoint/output aliasing is safe for the same disjoint-pair reason.
    pair_count = before.shape[1] // 2
    block_pairs = 256
    program_count = triton.cdiv(pair_count * before.shape[0], block_pairs)
    partial_gradients = torch.empty(
        program_count, dtype=torch.float32, device=before.device
    )
    _complex64_local_1q_vjp_adjoint_kernel[(program_count,)](
        torch.view_as_real(before),
        torch.view_as_real(adjoint),
        torch.view_as_real(matrix),
        torch.view_as_real(derivative_matrix),
        torch.view_as_real(next_adjoint),
        partial_gradients,
        pair_count,
        before.shape[0],
        before.stride(0),
        int(bit_position),
        REVERSE_KET=False,
        BLOCK_PAIRS=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return next_adjoint, partial_gradients.sum()


def fused_complex64_local_1q_reversible_vjp(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative_matrix: torch.Tensor,
    *,
    bit_position: int,
) -> torch.Tensor:
    """Reverse ket and bra and compute one parameter VJP in a single pass."""

    if (
        ket.device.type != "cuda"
        or ket.dtype != torch.complex64
        or ket.ndim != 2
        or not ket.is_contiguous()
        or adjoint.shape != ket.shape
        or adjoint.dtype != ket.dtype
        or adjoint.device != ket.device
        or not adjoint.is_contiguous()
        or adjoint.data_ptr() == ket.data_ptr()
    ):
        raise ValueError("fused reversible VJP requires distinct complex64 CUDA states")
    if matrix.shape != (2, 2) or derivative_matrix.shape != (2, 2):
        raise ValueError("fused reversible VJP requires 2x2 matrices")
    matrix = matrix.to(device=ket.device, dtype=ket.dtype).contiguous()
    derivative_matrix = derivative_matrix.to(
        device=ket.device, dtype=ket.dtype
    ).contiguous()
    pair_count = ket.shape[1] // 2
    block_pairs = 256
    program_count = triton.cdiv(pair_count * ket.shape[0], block_pairs)
    partial_gradients = torch.empty(
        program_count, dtype=torch.float32, device=ket.device
    )
    _complex64_local_1q_vjp_adjoint_kernel[(program_count,)](
        torch.view_as_real(ket),
        torch.view_as_real(adjoint),
        torch.view_as_real(matrix),
        torch.view_as_real(derivative_matrix),
        torch.view_as_real(adjoint),
        partial_gradients,
        pair_count,
        ket.shape[0],
        ket.stride(0),
        int(bit_position),
        REVERSE_KET=True,
        BLOCK_PAIRS=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return partial_gradients.sum()


@triton.jit(do_not_specialize=["rank_basis"])
def _complex64_sharded_1q_vjp_adjoint_kernel(
    local_before_parts,
    remote_before_parts,
    local_adjoint_parts,
    remote_adjoint_parts,
    matrix_parts,
    derivative_parts,
    next_adjoint_parts,
    partial_gradients,
    element_count,
    rank_basis,
    BLOCK: tl.constexpr,  # noqa: N803
):
    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    mask = offset < element_count
    local_before_real = tl.load(local_before_parts + 2 * offset, mask=mask, other=0.0)
    local_before_imag = tl.load(
        local_before_parts + 2 * offset + 1, mask=mask, other=0.0
    )
    remote_before_real = tl.load(remote_before_parts + 2 * offset, mask=mask, other=0.0)
    remote_before_imag = tl.load(
        remote_before_parts + 2 * offset + 1, mask=mask, other=0.0
    )
    local_adjoint_real = tl.load(local_adjoint_parts + 2 * offset, mask=mask, other=0.0)
    local_adjoint_imag = tl.load(
        local_adjoint_parts + 2 * offset + 1, mask=mask, other=0.0
    )
    remote_adjoint_real = tl.load(
        remote_adjoint_parts + 2 * offset, mask=mask, other=0.0
    )
    remote_adjoint_imag = tl.load(
        remote_adjoint_parts + 2 * offset + 1, mask=mask, other=0.0
    )

    before0_real = tl.where(rank_basis == 0, local_before_real, remote_before_real)
    before0_imag = tl.where(rank_basis == 0, local_before_imag, remote_before_imag)
    before1_real = tl.where(rank_basis == 0, remote_before_real, local_before_real)
    before1_imag = tl.where(rank_basis == 0, remote_before_imag, local_before_imag)
    adjoint0_real = tl.where(rank_basis == 0, local_adjoint_real, remote_adjoint_real)
    adjoint0_imag = tl.where(rank_basis == 0, local_adjoint_imag, remote_adjoint_imag)
    adjoint1_real = tl.where(rank_basis == 0, remote_adjoint_real, local_adjoint_real)
    adjoint1_imag = tl.where(rank_basis == 0, remote_adjoint_imag, local_adjoint_imag)

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
    tl.store(next_adjoint_parts + 2 * offset, next_real, mask=mask)
    tl.store(next_adjoint_parts + 2 * offset + 1, next_imag, mask=mask)


def fused_complex64_sharded_1q_vjp_adjoint(
    local_before: torch.Tensor,
    remote_before: torch.Tensor,
    local_adjoint: torch.Tensor,
    remote_adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative_matrix: torch.Tensor,
    *,
    rank_basis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse one rank's sharded-1q VJP and inverse-adjoint output."""

    tensors = (local_before, remote_before, local_adjoint, remote_adjoint)
    if (
        any(tensor.device.type != "cuda" for tensor in tensors)
        or any(tensor.dtype != torch.complex64 for tensor in tensors)
        or any(tensor.shape != local_before.shape for tensor in tensors)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or rank_basis not in (0, 1)
    ):
        raise ValueError(
            "sharded Triton VJP requires matching contiguous complex64 chunks"
        )
    matrix = matrix.to(device=local_before.device, dtype=torch.complex64).contiguous()
    derivative_matrix = derivative_matrix.to(
        device=local_before.device, dtype=torch.complex64
    ).contiguous()
    next_adjoint = torch.empty_like(local_adjoint)
    element_count = local_before.numel()
    block = 256
    program_count = triton.cdiv(element_count, block)
    partial_gradients = torch.empty(
        program_count, dtype=torch.float32, device=local_before.device
    )
    _complex64_sharded_1q_vjp_adjoint_kernel[(program_count,)](
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


__all__ = [
    "apply_complex64_local_1q",
    "fused_complex64_local_1q_vjp_adjoint",
    "fused_complex64_local_1q_reversible_vjp",
    "fused_complex64_sharded_1q_vjp_adjoint",
]
