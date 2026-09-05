"""Native Triton kernels for flat statevector gate application."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["bit_position", "compressed_start"])
def _complex64_control_one_pack_kernel(
    state_parts,
    packed_parts,
    chunk_count,
    batch_count,
    state_batch_stride,
    bit_position,
    compressed_start,
    BLOCK: tl.constexpr,  # noqa: N803
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = chunk_count * batch_count
    mask = linear < total
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
    real = tl.load(state_parts + 2 * state_index, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * state_index + 1, mask=mask, other=0.0)
    tl.store(packed_parts + 2 * linear, real, mask=mask)
    tl.store(packed_parts + 2 * linear + 1, imag, mask=mask)


@triton.jit(do_not_specialize=["bit_position", "compressed_start"])
def _complex64_control_one_unpack_kernel(
    packed_parts,
    output_parts,
    chunk_count,
    batch_count,
    output_batch_stride,
    bit_position,
    compressed_start,
    BLOCK: tl.constexpr,  # noqa: N803
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = chunk_count * batch_count
    mask = linear < total
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
    real = tl.load(packed_parts + 2 * linear, mask=mask, other=0.0)
    imag = tl.load(packed_parts + 2 * linear + 1, mask=mask, other=0.0)
    tl.store(output_parts + 2 * output_index, real, mask=mask)
    tl.store(output_parts + 2 * output_index + 1, imag, mask=mask)


def pack_complex64_control_one(
    state: torch.Tensor,
    *,
    bit_position: int,
    compressed_start: int,
    compressed_end: int,
) -> torch.Tensor:
    """Pack the control-one subspace without materializing address indices."""

    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
    ):
        raise ValueError("Triton CX pack requires contiguous CUDA complex64 [B, N]")
    chunk_count = compressed_end - compressed_start
    if chunk_count <= 0 or not 0 <= bit_position < state.shape[1].bit_length() - 1:
        raise ValueError("invalid Triton CX pack range or bit position")
    packed = torch.empty(
        (state.shape[0], chunk_count), dtype=state.dtype, device=state.device
    )
    block = 256
    _complex64_control_one_pack_kernel[
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


def unpack_complex64_control_one(
    packed: torch.Tensor,
    output: torch.Tensor,
    *,
    bit_position: int,
    compressed_start: int,
) -> None:
    """Scatter a packed control-one subspace without address-index tensors."""

    if (
        packed.device.type != "cuda"
        or packed.dtype != torch.complex64
        or packed.ndim != 2
        or not packed.is_contiguous()
        or output.device != packed.device
        or output.dtype != packed.dtype
        or output.ndim != 2
        or not output.is_contiguous()
        or output.shape[0] != packed.shape[0]
    ):
        raise ValueError("Triton CX unpack requires matching CUDA complex64 tensors")
    chunk_count = packed.shape[1]
    block = 256
    _complex64_control_one_unpack_kernel[
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


@triton.jit
def _single_qubit_matrix_kernel(
    state_parts,
    matrix_parts,
    output_parts,
    pair_count,
    pairs_per_batch: tl.constexpr,
    amplitudes_per_batch: tl.constexpr,
    target_mask: tl.constexpr,
    matrix_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    pairs = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = pairs < pair_count
    batch = pairs // pairs_per_batch
    local_pair = pairs - batch * pairs_per_batch
    lower = local_pair & (target_mask - 1)
    zero_amplitude = ((local_pair - lower) << 1) | lower
    zero = batch * amplitudes_per_batch + zero_amplitude
    one = zero + target_mask
    real0 = tl.load(state_parts + 2 * zero, mask=valid, other=0.0)
    imag0 = tl.load(state_parts + 2 * zero + 1, mask=valid, other=0.0)
    real1 = tl.load(state_parts + 2 * one, mask=valid, other=0.0)
    imag1 = tl.load(state_parts + 2 * one + 1, mask=valid, other=0.0)
    matrix_base = 2 * batch * matrix_batch_stride
    m00r = tl.load(matrix_parts + matrix_base)
    m00i = tl.load(matrix_parts + matrix_base + 1)
    m01r = tl.load(matrix_parts + matrix_base + 2)
    m01i = tl.load(matrix_parts + matrix_base + 3)
    m10r = tl.load(matrix_parts + matrix_base + 4)
    m10i = tl.load(matrix_parts + matrix_base + 5)
    m11r = tl.load(matrix_parts + matrix_base + 6)
    m11i = tl.load(matrix_parts + matrix_base + 7)
    out0r = m00r * real0 - m00i * imag0 + m01r * real1 - m01i * imag1
    out0i = m00r * imag0 + m00i * real0 + m01r * imag1 + m01i * real1
    out1r = m10r * real0 - m10i * imag0 + m11r * real1 - m11i * imag1
    out1i = m10r * imag0 + m10i * real0 + m11r * imag1 + m11i * real1
    tl.store(output_parts + 2 * zero, out0r, mask=valid)
    tl.store(output_parts + 2 * zero + 1, out0i, mask=valid)
    tl.store(output_parts + 2 * one, out1r, mask=valid)
    tl.store(output_parts + 2 * one + 1, out1i, mask=valid)


@triton.jit
def _single_qubit_matrix_backward_kernel(
    state_parts,
    gradient_parts,
    adjoint_parts,
    state_gradient_parts,
    partial_parts,
    pairs_per_batch: tl.constexpr,
    amplitudes_per_batch: tl.constexpr,
    target_mask: tl.constexpr,
    matrix_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    local_pair = block * block_size + tl.arange(0, block_size)
    valid = local_pair < pairs_per_batch
    lower = local_pair & (target_mask - 1)
    zero_amplitude = ((local_pair - lower) << 1) | lower
    zero = batch * amplitudes_per_batch + zero_amplitude
    one = zero + target_mask
    x0r = tl.load(state_parts + 2 * zero, mask=valid, other=0.0)
    x0i = tl.load(state_parts + 2 * zero + 1, mask=valid, other=0.0)
    x1r = tl.load(state_parts + 2 * one, mask=valid, other=0.0)
    x1i = tl.load(state_parts + 2 * one + 1, mask=valid, other=0.0)
    g0r = tl.load(gradient_parts + 2 * zero, mask=valid, other=0.0)
    g0i = tl.load(gradient_parts + 2 * zero + 1, mask=valid, other=0.0)
    g1r = tl.load(gradient_parts + 2 * one, mask=valid, other=0.0)
    g1i = tl.load(gradient_parts + 2 * one + 1, mask=valid, other=0.0)
    base = 2 * batch * matrix_batch_stride
    a00r = tl.load(adjoint_parts + base)
    a00i = tl.load(adjoint_parts + base + 1)
    a01r = tl.load(adjoint_parts + base + 2)
    a01i = tl.load(adjoint_parts + base + 3)
    a10r = tl.load(adjoint_parts + base + 4)
    a10i = tl.load(adjoint_parts + base + 5)
    a11r = tl.load(adjoint_parts + base + 6)
    a11i = tl.load(adjoint_parts + base + 7)
    dx0r = a00r * g0r - a00i * g0i + a01r * g1r - a01i * g1i
    dx0i = a00r * g0i + a00i * g0r + a01r * g1i + a01i * g1r
    dx1r = a10r * g0r - a10i * g0i + a11r * g1r - a11i * g1i
    dx1i = a10r * g0i + a10i * g0r + a11r * g1i + a11i * g1r
    tl.store(state_gradient_parts + 2 * zero, dx0r, mask=valid)
    tl.store(state_gradient_parts + 2 * zero + 1, dx0i, mask=valid)
    tl.store(state_gradient_parts + 2 * one, dx1r, mask=valid)
    tl.store(state_gradient_parts + 2 * one + 1, dx1i, mask=valid)
    partial = 8 * batch
    tl.atomic_add(partial_parts + partial, tl.sum(g0r * x0r + g0i * x0i))
    tl.atomic_add(partial_parts + partial + 1, tl.sum(g0i * x0r - g0r * x0i))
    tl.atomic_add(partial_parts + partial + 2, tl.sum(g0r * x1r + g0i * x1i))
    tl.atomic_add(partial_parts + partial + 3, tl.sum(g0i * x1r - g0r * x1i))
    tl.atomic_add(partial_parts + partial + 4, tl.sum(g1r * x0r + g1i * x0i))
    tl.atomic_add(partial_parts + partial + 5, tl.sum(g1i * x0r - g1r * x0i))
    tl.atomic_add(partial_parts + partial + 6, tl.sum(g1r * x1r + g1i * x1i))
    tl.atomic_add(partial_parts + partial + 7, tl.sum(g1i * x1r - g1r * x1i))


@triton.jit
def _cx_sequence_kernel(
    state_parts,
    output_parts,
    control_masks,
    target_masks,
    element_count,
    amplitudes_per_batch: tl.constexpr,
    sequence_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < element_count
    batch = offsets // amplitudes_per_batch
    source_amplitude = offsets - batch * amplitudes_per_batch
    for reverse_index in tl.static_range(0, sequence_length):
        index = sequence_length - reverse_index - 1
        control_mask = tl.load(control_masks + index)
        target_mask = tl.load(target_masks + index)
        source_amplitude = tl.where(
            (source_amplitude & control_mask) != 0,
            source_amplitude ^ target_mask,
            source_amplitude,
        )
    source = batch * amplitudes_per_batch + source_amplitude
    real = tl.load(state_parts + 2 * source, mask=valid, other=0.0)
    imag = tl.load(state_parts + 2 * source + 1, mask=valid, other=0.0)
    tl.store(output_parts + 2 * offsets, real, mask=valid)
    tl.store(output_parts + 2 * offsets + 1, imag, mask=valid)


@triton.jit
def _ry_rz_pair_kernel(
    state_parts,
    ry_angles,
    rz_angles,
    output_parts,
    pair_count,
    pairs_per_batch: tl.constexpr,
    amplitudes_per_batch: tl.constexpr,
    target_mask: tl.constexpr,
    angle_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    pairs = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = pairs < pair_count
    batch = pairs // pairs_per_batch
    local_pair = pairs - batch * pairs_per_batch
    lower = local_pair & (target_mask - 1)
    zero_amplitude = ((local_pair - lower) << 1) | lower
    zero = batch * amplitudes_per_batch + zero_amplitude
    one = zero + target_mask
    real0 = tl.load(state_parts + 2 * zero, mask=valid, other=0.0)
    imag0 = tl.load(state_parts + 2 * zero + 1, mask=valid, other=0.0)
    real1 = tl.load(state_parts + 2 * one, mask=valid, other=0.0)
    imag1 = tl.load(state_parts + 2 * one + 1, mask=valid, other=0.0)
    half_ry = 0.5 * tl.load(ry_angles + batch * angle_batch_stride)
    sin_ry, cos_ry = tl.sin(half_ry), tl.cos(half_ry)
    next_real0 = cos_ry * real0 - sin_ry * real1
    next_imag0 = cos_ry * imag0 - sin_ry * imag1
    next_real1 = sin_ry * real0 + cos_ry * real1
    next_imag1 = sin_ry * imag0 + cos_ry * imag1
    half_rz = 0.5 * tl.load(rz_angles + batch * angle_batch_stride)
    sin_rz, cos_rz = tl.sin(half_rz), tl.cos(half_rz)
    real0 = cos_rz * next_real0 + sin_rz * next_imag0
    imag0 = cos_rz * next_imag0 - sin_rz * next_real0
    real1 = cos_rz * next_real1 - sin_rz * next_imag1
    imag1 = cos_rz * next_imag1 + sin_rz * next_real1
    tl.store(output_parts + 2 * zero, real0, mask=valid)
    tl.store(output_parts + 2 * zero + 1, imag0, mask=valid)
    tl.store(output_parts + 2 * one, real1, mask=valid)
    tl.store(output_parts + 2 * one + 1, imag1, mask=valid)


def ry_rz_pair(
    state: torch.Tensor,
    ry_angles: torch.Tensor,
    rz_angles: torch.Tensor,
    *,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply one RY then RZ pair with a single flat-state CUDA kernel."""

    if not state.is_cuda or state.dtype != torch.complex64:
        raise ValueError("ry_rz_pair requires a CUDA complex64 state")
    if ry_angles.shape != rz_angles.shape or ry_angles.shape != (state.shape[0], 1):
        raise ValueError("RY/RZ angles must have shape [batch, 1]")
    if not state.is_contiguous():
        state = state.contiguous()
    ry_angles = ry_angles.contiguous()
    rz_angles = rz_angles.contiguous()
    output_parts = torch.empty(
        *state.shape, 2, dtype=torch.float32, device=state.device
    )
    pairs_per_batch = 2 ** (n_wires - 1)
    pair_count = state.shape[0] * pairs_per_batch
    block_size = 256
    _ry_rz_pair_kernel[(triton.cdiv(pair_count, block_size),)](
        torch.view_as_real(state),
        ry_angles,
        rz_angles,
        output_parts,
        pair_count,
        pairs_per_batch=pairs_per_batch,
        amplitudes_per_batch=2**n_wires,
        target_mask=1 << (n_wires - 1 - wire),
        angle_batch_stride=ry_angles.stride(0),
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    return torch.view_as_complex(output_parts)


def _launch_cx_sequence(
    state: torch.Tensor,
    control_masks: torch.Tensor,
    target_masks: torch.Tensor,
    n_wires: int,
) -> torch.Tensor:
    if not state.is_contiguous():
        state = state.contiguous()
    output_parts = torch.empty(
        *state.shape, 2, dtype=torch.float32, device=state.device
    )
    element_count = state.numel()
    block_size = 256
    _cx_sequence_kernel[(triton.cdiv(element_count, block_size),)](
        torch.view_as_real(state),
        output_parts,
        control_masks,
        target_masks,
        element_count,
        amplitudes_per_batch=2**n_wires,
        sequence_length=control_masks.numel(),
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    return torch.view_as_complex(output_parts)


def _launch_single_qubit_matrix(
    state: torch.Tensor, matrix: torch.Tensor, wire: int, n_wires: int
) -> torch.Tensor:
    if not state.is_contiguous():
        state = state.contiguous()
    matrix = matrix.contiguous()
    output_parts = torch.empty(
        *state.shape, 2, dtype=torch.float32, device=state.device
    )
    pairs_per_batch = 2 ** (n_wires - 1)
    pair_count = state.shape[0] * pairs_per_batch
    block_size = 256
    _single_qubit_matrix_kernel[(triton.cdiv(pair_count, block_size),)](
        torch.view_as_real(state),
        torch.view_as_real(matrix),
        output_parts,
        pair_count,
        pairs_per_batch=pairs_per_batch,
        amplitudes_per_batch=2**n_wires,
        target_mask=1 << (n_wires - 1 - wire),
        matrix_batch_stride=0 if matrix.ndim == 2 else matrix.stride(0),
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    return torch.view_as_complex(output_parts)


def _launch_single_qubit_matrix_backward(
    state: torch.Tensor,
    gradient: torch.Tensor,
    matrix: torch.Tensor,
    wire: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = gradient.contiguous()
    adjoint = matrix.conj().transpose(-2, -1).contiguous()
    state_gradient_parts = torch.empty(
        *state.shape, 2, dtype=torch.float32, device=state.device
    )
    pairs_per_batch = 2 ** (n_wires - 1)
    block_size = 1024
    blocks_per_batch = triton.cdiv(pairs_per_batch, block_size)
    partial_parts = torch.zeros(
        state.shape[0],
        2,
        2,
        2,
        dtype=torch.float32,
        device=state.device,
    )
    _single_qubit_matrix_backward_kernel[(blocks_per_batch, state.shape[0])](
        torch.view_as_real(state),
        torch.view_as_real(gradient),
        torch.view_as_real(adjoint),
        state_gradient_parts,
        partial_parts,
        pairs_per_batch=pairs_per_batch,
        amplitudes_per_batch=2**n_wires,
        target_mask=1 << (n_wires - 1 - wire),
        matrix_batch_stride=0 if matrix.ndim == 2 else matrix.stride(0),
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    matrix_gradient = torch.view_as_complex(partial_parts)
    if matrix.ndim == 2:
        matrix_gradient = matrix_gradient.sum(dim=0)
    return torch.view_as_complex(state_gradient_parts), matrix_gradient


class _SingleQubitMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, matrix, wire, n_wires):
        ctx.save_for_backward(state, matrix)
        ctx.wire = int(wire)
        ctx.n_wires = int(n_wires)
        return _launch_single_qubit_matrix(state, matrix, ctx.wire, ctx.n_wires)

    @staticmethod
    def backward(ctx, gradient):
        state, matrix = ctx.saved_tensors
        gradient = gradient.resolve_conj().resolve_neg()
        state_gradient, matrix_gradient = _launch_single_qubit_matrix_backward(
            state,
            gradient,
            matrix,
            ctx.wire,
            ctx.n_wires,
        )
        return state_gradient, matrix_gradient, None, None


def single_qubit_matrix(
    state: torch.Tensor, matrix: torch.Tensor, *, wire: int, n_wires: int
) -> torch.Tensor:
    """Apply a 2x2 matrix with a differentiable flat-state CUDA kernel."""

    if not state.is_cuda or state.dtype != torch.complex64:
        raise ValueError("single_qubit_matrix requires a CUDA complex64 state")
    if matrix.shape not in {(2, 2), (state.shape[0], 2, 2)}:
        raise ValueError("matrix must have shape [2, 2] or [batch, 2, 2]")
    return _SingleQubitMatrix.apply(state, matrix, wire, n_wires)


class _CXSequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state,
        control_masks,
        target_masks,
        reverse_control_masks,
        reverse_target_masks,
        n_wires,
    ):
        ctx.save_for_backward(reverse_control_masks, reverse_target_masks)
        ctx.n_wires = int(n_wires)
        return _launch_cx_sequence(state, control_masks, target_masks, ctx.n_wires)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.resolve_conj().resolve_neg()
        reverse_control_masks, reverse_target_masks = ctx.saved_tensors
        return (
            _launch_cx_sequence(
                gradient,
                reverse_control_masks,
                reverse_target_masks,
                ctx.n_wires,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def cx_sequence(
    state: torch.Tensor,
    *,
    control_masks: torch.Tensor,
    target_masks: torch.Tensor,
    reverse_control_masks: torch.Tensor,
    reverse_target_masks: torch.Tensor,
    n_wires: int,
) -> torch.Tensor:
    """Apply a sequence of CNOT gates with one flat-state CUDA kernel."""

    if not state.is_cuda or state.dtype != torch.complex64:
        raise ValueError("cx_sequence requires a CUDA complex64 state")
    if control_masks.shape != target_masks.shape or control_masks.numel() < 2:
        raise ValueError("cx_sequence requires at least two matched gates")
    return _CXSequence.apply(
        state,
        control_masks,
        target_masks,
        reverse_control_masks,
        reverse_target_masks,
        n_wires,
    )


__all__ = [
    "cx_sequence",
    "pack_complex64_control_one",
    "ry_rz_pair",
    "single_qubit_matrix",
    "unpack_complex64_control_one",
]
