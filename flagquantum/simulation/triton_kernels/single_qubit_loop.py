"""Persistent Triton kernel for repeated local single-qubit gate loops."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rx_rz_loop_kernel(
    state_parts,
    rx_angles,
    rz_angles,
    output_parts,
    pair_count,
    depth,
    state_batch_stride: tl.constexpr,
    angle_batch_stride: tl.constexpr,
    block_pairs: tl.constexpr,
):
    batch = tl.program_id(1)
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = pairs < pair_count
    base = batch * state_batch_stride + pairs * 2
    real0 = tl.load(state_parts + 2 * base, mask=mask, other=0.0)
    imag0 = tl.load(state_parts + 2 * base + 1, mask=mask, other=0.0)
    real1 = tl.load(state_parts + 2 * (base + 1), mask=mask, other=0.0)
    imag1 = tl.load(state_parts + 2 * (base + 1) + 1, mask=mask, other=0.0)
    for layer in range(0, depth):
        angle_offset = batch * angle_batch_stride + layer
        half_rx = 0.5 * tl.load(rx_angles + angle_offset)
        sin_rx, cos_rx = tl.sin(half_rx), tl.cos(half_rx)
        next_real0 = cos_rx * real0 + sin_rx * imag1
        next_imag0 = cos_rx * imag0 - sin_rx * real1
        next_real1 = sin_rx * imag0 + cos_rx * real1
        next_imag1 = -sin_rx * real0 + cos_rx * imag1
        half_rz = 0.5 * tl.load(rz_angles + angle_offset)
        sin_rz, cos_rz = tl.sin(half_rz), tl.cos(half_rz)
        real0 = cos_rz * next_real0 + sin_rz * next_imag0
        imag0 = cos_rz * next_imag0 - sin_rz * next_real0
        real1 = cos_rz * next_real1 - sin_rz * next_imag1
        imag1 = cos_rz * next_imag1 + sin_rz * next_real1
    tl.store(output_parts + 2 * base, real0, mask=mask)
    tl.store(output_parts + 2 * base + 1, imag0, mask=mask)
    tl.store(output_parts + 2 * (base + 1), real1, mask=mask)
    tl.store(output_parts + 2 * (base + 1) + 1, imag1, mask=mask)


@triton.jit
def _rx_rz_loop_tangent_kernel(
    state_parts,
    rx_angles,
    rz_angles,
    tangent_parts,
    pair_count,
    depth: tl.constexpr,
    state_batch_stride: tl.constexpr,
    angle_batch_stride: tl.constexpr,
    tangent_parameter_stride: tl.constexpr,
    tangent_batch_stride: tl.constexpr,
    block_pairs: tl.constexpr,
):
    """Propagate one exact parameter tangent per program-grid plane."""

    parameter = tl.program_id(2)
    batch = tl.program_id(1)
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = pairs < pair_count
    base = batch * state_batch_stride + pairs * 2
    real0 = tl.load(state_parts + 2 * base, mask=mask, other=0.0)
    imag0 = tl.load(state_parts + 2 * base + 1, mask=mask, other=0.0)
    real1 = tl.load(state_parts + 2 * (base + 1), mask=mask, other=0.0)
    imag1 = tl.load(state_parts + 2 * (base + 1) + 1, mask=mask, other=0.0)
    tangent_real0 = tl.zeros((block_pairs,), tl.float32)
    tangent_imag0 = tl.zeros((block_pairs,), tl.float32)
    tangent_real1 = tl.zeros((block_pairs,), tl.float32)
    tangent_imag1 = tl.zeros((block_pairs,), tl.float32)
    target_layer = parameter // 2
    target_is_rz = parameter % 2
    for layer in range(0, depth):
        angle_offset = batch * angle_batch_stride + layer
        half_rx = 0.5 * tl.load(rx_angles + angle_offset)
        sin_rx, cos_rx = tl.sin(half_rx), tl.cos(half_rx)
        next_real0 = cos_rx * real0 + sin_rx * imag1
        next_imag0 = cos_rx * imag0 - sin_rx * real1
        next_real1 = sin_rx * imag0 + cos_rx * real1
        next_imag1 = -sin_rx * real0 + cos_rx * imag1
        next_tangent_real0 = cos_rx * tangent_real0 + sin_rx * tangent_imag1
        next_tangent_imag0 = cos_rx * tangent_imag0 - sin_rx * tangent_real1
        next_tangent_real1 = sin_rx * tangent_imag0 + cos_rx * tangent_real1
        next_tangent_imag1 = -sin_rx * tangent_real0 + cos_rx * tangent_imag1
        is_rx_target = (layer == target_layer) & (target_is_rz == 0)
        next_tangent_real0 += tl.where(
            is_rx_target, 0.5 * (-sin_rx * real0 + cos_rx * imag1), 0.0
        )
        next_tangent_imag0 += tl.where(
            is_rx_target, 0.5 * (-sin_rx * imag0 - cos_rx * real1), 0.0
        )
        next_tangent_real1 += tl.where(
            is_rx_target, 0.5 * (cos_rx * imag0 - sin_rx * real1), 0.0
        )
        next_tangent_imag1 += tl.where(
            is_rx_target, 0.5 * (-cos_rx * real0 - sin_rx * imag1), 0.0
        )
        half_rz = 0.5 * tl.load(rz_angles + angle_offset)
        sin_rz, cos_rz = tl.sin(half_rz), tl.cos(half_rz)
        real0 = cos_rz * next_real0 + sin_rz * next_imag0
        imag0 = cos_rz * next_imag0 - sin_rz * next_real0
        real1 = cos_rz * next_real1 - sin_rz * next_imag1
        imag1 = cos_rz * next_imag1 + sin_rz * next_real1
        tangent_real0 = cos_rz * next_tangent_real0 + sin_rz * next_tangent_imag0
        tangent_imag0 = cos_rz * next_tangent_imag0 - sin_rz * next_tangent_real0
        tangent_real1 = cos_rz * next_tangent_real1 - sin_rz * next_tangent_imag1
        tangent_imag1 = cos_rz * next_tangent_imag1 + sin_rz * next_tangent_real1
        is_rz_target = (layer == target_layer) & (target_is_rz == 1)
        tangent_real0 += tl.where(
            is_rz_target, 0.5 * (-sin_rz * next_real0 + cos_rz * next_imag0), 0.0
        )
        tangent_imag0 += tl.where(
            is_rz_target, 0.5 * (-sin_rz * next_imag0 - cos_rz * next_real0), 0.0
        )
        tangent_real1 += tl.where(
            is_rz_target, 0.5 * (-sin_rz * next_real1 - cos_rz * next_imag1), 0.0
        )
        tangent_imag1 += tl.where(
            is_rz_target, 0.5 * (-sin_rz * next_imag1 + cos_rz * next_real1), 0.0
        )
    tangent_base = (
        parameter * tangent_parameter_stride + batch * tangent_batch_stride + pairs * 2
    )
    tl.store(tangent_parts + 2 * tangent_base, tangent_real0, mask=mask)
    tl.store(tangent_parts + 2 * tangent_base + 1, tangent_imag0, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 1), tangent_real1, mask=mask)
    tl.store(
        tangent_parts + 2 * (tangent_base + 1) + 1,
        tangent_imag1,
        mask=mask,
    )


@triton.jit
def _rx_rz_loop_backward_kernel(
    output_parts,
    gradient_parts,
    rx_angles,
    rz_angles,
    state_gradient_parts,
    rx_gradients,
    rz_gradients,
    pair_count,
    depth,
    state_batch_stride: tl.constexpr,
    angle_batch_stride: tl.constexpr,
    block_pairs: tl.constexpr,
):
    batch = tl.program_id(1)
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = pairs < pair_count
    base = batch * state_batch_stride + pairs * 2
    real0 = tl.load(output_parts + 2 * base, mask=mask, other=0.0)
    imag0 = tl.load(output_parts + 2 * base + 1, mask=mask, other=0.0)
    real1 = tl.load(output_parts + 2 * (base + 1), mask=mask, other=0.0)
    imag1 = tl.load(output_parts + 2 * (base + 1) + 1, mask=mask, other=0.0)
    grad_real0 = tl.load(gradient_parts + 2 * base, mask=mask, other=0.0)
    grad_imag0 = tl.load(gradient_parts + 2 * base + 1, mask=mask, other=0.0)
    grad_real1 = tl.load(gradient_parts + 2 * (base + 1), mask=mask, other=0.0)
    grad_imag1 = tl.load(gradient_parts + 2 * (base + 1) + 1, mask=mask, other=0.0)
    for reverse_layer in range(0, depth):
        layer = depth - reverse_layer - 1
        angle_offset = batch * angle_batch_stride + layer
        rz_contribution = 0.5 * (
            grad_real0 * imag0
            - grad_imag0 * real0
            - grad_real1 * imag1
            + grad_imag1 * real1
        )
        tl.atomic_add(rz_gradients + angle_offset, tl.sum(rz_contribution))
        half_rz = 0.5 * tl.load(rz_angles + angle_offset)
        sin_rz, cos_rz = tl.sin(half_rz), tl.cos(half_rz)
        rx_real0 = cos_rz * real0 - sin_rz * imag0
        rx_imag0 = sin_rz * real0 + cos_rz * imag0
        rx_real1 = cos_rz * real1 + sin_rz * imag1
        rx_imag1 = -sin_rz * real1 + cos_rz * imag1
        rx_grad_real0 = cos_rz * grad_real0 - sin_rz * grad_imag0
        rx_grad_imag0 = sin_rz * grad_real0 + cos_rz * grad_imag0
        rx_grad_real1 = cos_rz * grad_real1 + sin_rz * grad_imag1
        rx_grad_imag1 = -sin_rz * grad_real1 + cos_rz * grad_imag1
        rx_contribution = 0.5 * (
            rx_grad_real0 * rx_imag1
            - rx_grad_imag0 * rx_real1
            + rx_grad_real1 * rx_imag0
            - rx_grad_imag1 * rx_real0
        )
        tl.atomic_add(rx_gradients + angle_offset, tl.sum(rx_contribution))
        half_rx = 0.5 * tl.load(rx_angles + angle_offset)
        sin_rx, cos_rx = tl.sin(half_rx), tl.cos(half_rx)
        real0 = cos_rx * rx_real0 - sin_rx * rx_imag1
        imag0 = cos_rx * rx_imag0 + sin_rx * rx_real1
        real1 = -sin_rx * rx_imag0 + cos_rx * rx_real1
        imag1 = sin_rx * rx_real0 + cos_rx * rx_imag1
        grad_real0 = cos_rx * rx_grad_real0 - sin_rx * rx_grad_imag1
        grad_imag0 = cos_rx * rx_grad_imag0 + sin_rx * rx_grad_real1
        grad_real1 = -sin_rx * rx_grad_imag0 + cos_rx * rx_grad_real1
        grad_imag1 = sin_rx * rx_grad_real0 + cos_rx * rx_grad_imag1
    tl.store(state_gradient_parts + 2 * base, grad_real0, mask=mask)
    tl.store(state_gradient_parts + 2 * base + 1, grad_imag0, mask=mask)
    tl.store(state_gradient_parts + 2 * (base + 1), grad_real1, mask=mask)
    tl.store(state_gradient_parts + 2 * (base + 1) + 1, grad_imag1, mask=mask)


def _launch(
    state: torch.Tensor, rx_angles: torch.Tensor, rz_angles: torch.Tensor
) -> torch.Tensor:
    output_parts = torch.empty(
        *state.shape, 2, dtype=torch.float32, device=state.device
    )
    batch, pair_count, _ = state.shape
    block_pairs = 256
    grid = (triton.cdiv(pair_count, block_pairs), batch)
    _rx_rz_loop_kernel[grid](
        torch.view_as_real(state),
        rx_angles,
        rz_angles,
        output_parts,
        pair_count,
        int(rx_angles.shape[1]),
        state_batch_stride=state.stride(0),
        angle_batch_stride=rx_angles.stride(0),
        block_pairs=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return torch.view_as_complex(output_parts)


class _RepeatedRXRZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, rx_angles, rz_angles):
        output = _launch(state, rx_angles, rz_angles)
        ctx.save_for_backward(output, rx_angles, rz_angles)
        return output

    @staticmethod
    def backward(ctx, gradient):
        output, rx_angles, rz_angles = ctx.saved_tensors
        gradient = gradient.resolve_conj().resolve_neg().contiguous()
        state_gradient_parts = torch.empty(
            *output.shape, 2, dtype=torch.float32, device=output.device
        )
        rx_gradients = torch.zeros_like(rx_angles)
        rz_gradients = torch.zeros_like(rz_angles)
        batch, pair_count, _ = output.shape
        block_pairs = 256
        grid = (triton.cdiv(pair_count, block_pairs), batch)
        _rx_rz_loop_backward_kernel[grid](
            torch.view_as_real(output),
            torch.view_as_real(gradient),
            rx_angles,
            rz_angles,
            state_gradient_parts,
            rx_gradients,
            rz_gradients,
            pair_count,
            int(rx_angles.shape[1]),
            state_batch_stride=output.stride(0),
            angle_batch_stride=rx_angles.stride(0),
            block_pairs=block_pairs,
            num_warps=8,
            num_stages=2,
        )
        return torch.view_as_complex(state_gradient_parts), rx_gradients, rz_gradients


def repeated_rx_rz(
    state: torch.Tensor, rx_angles: torch.Tensor, rz_angles: torch.Tensor
) -> torch.Tensor:
    """Apply repeated RX then RZ gates to `[batch, pairs, 2]` amplitudes."""

    if state.ndim != 3 or int(state.shape[-1]) != 2:
        raise ValueError("state must have shape [batch, pairs, 2]")
    if rx_angles.shape != rz_angles.shape or rx_angles.ndim != 2:
        raise ValueError("RX/RZ angles must have matching [batch, depth] shapes")
    if int(rx_angles.shape[0]) != int(state.shape[0]):
        raise ValueError("angle batch must match state batch")
    if not state.is_cuda or state.dtype != torch.complex64:
        output = state
        for layer in range(int(rx_angles.shape[1])):
            rx = rx_angles[:, layer].reshape(-1, 1, 1) / 2
            rz = rz_angles[:, layer].reshape(-1, 1, 1) / 2
            cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
            zero, one = output[..., :1], output[..., 1:]
            output = torch.cat(
                (cos_rx * zero - 1j * sin_rx * one, -1j * sin_rx * zero + cos_rx * one),
                dim=-1,
            )
            output = output * torch.cat(
                (torch.exp(-1j * rz), torch.exp(1j * rz)), dim=-1
            )
        return output
    state = state.contiguous()
    rx_angles = rx_angles.contiguous()
    rz_angles = rz_angles.contiguous()
    return _RepeatedRXRZ.apply(state, rx_angles, rz_angles)


def repeated_rx_rz_tangents(
    state: torch.Tensor, rx_angles: torch.Tensor, rz_angles: torch.Tensor
) -> torch.Tensor:
    """Return exact RX/RZ parameter tangents as ``[2*depth,batch,pairs,2]``.

    Parameters are interleaved by layer as RX, RZ. The CUDA fast path is an
    experimental QNG building block; it does not claim support for entangling
    HVA regions.
    """

    if state.ndim != 3 or int(state.shape[-1]) != 2:
        raise ValueError("state must have shape [batch, pairs, 2]")
    if rx_angles.shape != rz_angles.shape or rx_angles.ndim != 2:
        raise ValueError("RX/RZ angles must have matching [batch, depth] shapes")
    if int(rx_angles.shape[0]) != int(state.shape[0]):
        raise ValueError("angle batch must match state batch")
    depth = int(rx_angles.shape[1])
    if not state.is_cuda or state.dtype != torch.complex64:
        tangents = []
        for layer in range(depth):
            for family in ("rx", "rz"):
                rx = rx_angles.detach().clone().requires_grad_(family == "rx")
                rz = rz_angles.detach().clone().requires_grad_(family == "rz")
                selected = rx if family == "rx" else rz

                def selected_output(value):
                    return repeated_rx_rz(
                        state,
                        value if family == "rx" else rx,
                        value if family == "rz" else rz,
                    )

                real_jacobian = torch.autograd.functional.jacobian(
                    lambda value: selected_output(value).real,
                    selected,
                    vectorize=True,
                )
                imag_jacobian = torch.autograd.functional.jacobian(
                    lambda value: selected_output(value).imag,
                    selected,
                    vectorize=True,
                )
                jacobian = torch.complex(real_jacobian, imag_jacobian)
                indices = torch.arange(int(state.shape[0]), device=state.device)
                tangents.append(jacobian[indices, :, :, indices, layer])
        return torch.stack(tangents)
    state = state.contiguous()
    rx_angles = rx_angles.contiguous()
    rz_angles = rz_angles.contiguous()
    batch, pair_count, _ = state.shape
    tangent_parts = torch.empty(
        2 * depth, batch, pair_count, 2, 2, dtype=torch.float32, device=state.device
    )
    block_pairs = 256
    grid = (triton.cdiv(pair_count, block_pairs), batch, 2 * depth)
    _rx_rz_loop_tangent_kernel[grid](
        torch.view_as_real(state),
        rx_angles,
        rz_angles,
        tangent_parts,
        pair_count,
        depth=depth,
        state_batch_stride=state.stride(0),
        angle_batch_stride=rx_angles.stride(0),
        tangent_parameter_stride=tangent_parts.stride(0) // 2,
        tangent_batch_stride=tangent_parts.stride(1) // 2,
        block_pairs=block_pairs,
        num_warps=8,
        num_stages=2,
    )
    return torch.view_as_complex(tangent_parts)


__all__ = ["repeated_rx_rz", "repeated_rx_rz_tangents"]
