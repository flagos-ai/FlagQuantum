"""Gate-aware forward-mode Triton runtime for bond-resolved Heisenberg HVA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _insert_zero_bit(values, position: tl.constexpr):
    lower_mask: tl.constexpr = (1 << position) - 1
    return (values & lower_mask) | ((values & ~lower_mask) << 1)


@triton.jit
def _two_qubit_forward_tangent_kernel(
    augmented_in,
    augmented_out,
    angle_ptr,
    dimension: tl.constexpr,
    lanes: tl.constexpr,
    parameter_lane: tl.constexpr,
    bit_left: tl.constexpr,
    bit_right: tl.constexpr,
    family: tl.constexpr,
    block_groups: tl.constexpr,
):
    lane = tl.program_id(1)
    groups = tl.program_id(0) * block_groups + tl.arange(0, block_groups)
    group_count: tl.constexpr = dimension // 4
    mask = groups < group_count
    low: tl.constexpr = min(bit_left, bit_right)
    high: tl.constexpr = max(bit_left, bit_right)
    base = _insert_zero_bit(_insert_zero_bit(groups, low), high)
    index0 = base
    index1 = base | (1 << bit_right)
    index2 = base | (1 << bit_left)
    index3 = base | (1 << bit_left) | (1 << bit_right)
    offset = lane * dimension
    r0 = tl.load(augmented_in + 2 * (offset + index0), mask=mask, other=0.0)
    i0 = tl.load(augmented_in + 2 * (offset + index0) + 1, mask=mask, other=0.0)
    r1 = tl.load(augmented_in + 2 * (offset + index1), mask=mask, other=0.0)
    i1 = tl.load(augmented_in + 2 * (offset + index1) + 1, mask=mask, other=0.0)
    r2 = tl.load(augmented_in + 2 * (offset + index2), mask=mask, other=0.0)
    i2 = tl.load(augmented_in + 2 * (offset + index2) + 1, mask=mask, other=0.0)
    r3 = tl.load(augmented_in + 2 * (offset + index3), mask=mask, other=0.0)
    i3 = tl.load(augmented_in + 2 * (offset + index3) + 1, mask=mask, other=0.0)
    if family == 0:
        pr0, pi0, pr1, pi1 = r3, i3, r2, i2
        pr2, pi2, pr3, pi3 = r1, i1, r0, i0
    elif family == 1:
        pr0, pi0, pr1, pi1 = -r3, -i3, r2, i2
        pr2, pi2, pr3, pi3 = r1, i1, -r0, -i0
    else:
        pr0, pi0, pr1, pi1 = r0, i0, -r1, -i1
        pr2, pi2, pr3, pi3 = -r2, -i2, r3, i3
    half = 0.5 * tl.load(angle_ptr)
    sine, cosine = tl.sin(half), tl.cos(half)
    nr0, ni0 = cosine * r0 + sine * pi0, cosine * i0 - sine * pr0
    nr1, ni1 = cosine * r1 + sine * pi1, cosine * i1 - sine * pr1
    nr2, ni2 = cosine * r2 + sine * pi2, cosine * i2 - sine * pr2
    nr3, ni3 = cosine * r3 + sine * pi3, cosine * i3 - sine * pr3
    selected = lane == parameter_lane
    state_r0 = tl.load(augmented_in + 2 * index0, mask=mask, other=0.0)
    state_i0 = tl.load(augmented_in + 2 * index0 + 1, mask=mask, other=0.0)
    state_r1 = tl.load(augmented_in + 2 * index1, mask=mask, other=0.0)
    state_i1 = tl.load(augmented_in + 2 * index1 + 1, mask=mask, other=0.0)
    state_r2 = tl.load(augmented_in + 2 * index2, mask=mask, other=0.0)
    state_i2 = tl.load(augmented_in + 2 * index2 + 1, mask=mask, other=0.0)
    state_r3 = tl.load(augmented_in + 2 * index3, mask=mask, other=0.0)
    state_i3 = tl.load(augmented_in + 2 * index3 + 1, mask=mask, other=0.0)
    if family == 0:
        spr0, spi0, spr1, spi1 = state_r3, state_i3, state_r2, state_i2
        spr2, spi2, spr3, spi3 = state_r1, state_i1, state_r0, state_i0
    elif family == 1:
        spr0, spi0, spr1, spi1 = -state_r3, -state_i3, state_r2, state_i2
        spr2, spi2, spr3, spi3 = state_r1, state_i1, -state_r0, -state_i0
    else:
        spr0, spi0, spr1, spi1 = state_r0, state_i0, -state_r1, -state_i1
        spr2, spi2, spr3, spi3 = -state_r2, -state_i2, state_r3, state_i3
    nr0 += tl.where(selected, 0.5 * (-sine * state_r0 + cosine * spi0), 0.0)
    ni0 += tl.where(selected, 0.5 * (-sine * state_i0 - cosine * spr0), 0.0)
    nr1 += tl.where(selected, 0.5 * (-sine * state_r1 + cosine * spi1), 0.0)
    ni1 += tl.where(selected, 0.5 * (-sine * state_i1 - cosine * spr1), 0.0)
    nr2 += tl.where(selected, 0.5 * (-sine * state_r2 + cosine * spi2), 0.0)
    ni2 += tl.where(selected, 0.5 * (-sine * state_i2 - cosine * spr2), 0.0)
    nr3 += tl.where(selected, 0.5 * (-sine * state_r3 + cosine * spi3), 0.0)
    ni3 += tl.where(selected, 0.5 * (-sine * state_i3 - cosine * spr3), 0.0)
    tl.store(augmented_out + 2 * (offset + index0), nr0, mask=mask)
    tl.store(augmented_out + 2 * (offset + index0) + 1, ni0, mask=mask)
    tl.store(augmented_out + 2 * (offset + index1), nr1, mask=mask)
    tl.store(augmented_out + 2 * (offset + index1) + 1, ni1, mask=mask)
    tl.store(augmented_out + 2 * (offset + index2), nr2, mask=mask)
    tl.store(augmented_out + 2 * (offset + index2) + 1, ni2, mask=mask)
    tl.store(augmented_out + 2 * (offset + index3), nr3, mask=mask)
    tl.store(augmented_out + 2 * (offset + index3) + 1, ni3, mask=mask)


@triton.jit
def _rz_forward_tangent_kernel(
    augmented_in,
    augmented_out,
    angle_ptr,
    dimension: tl.constexpr,
    lanes: tl.constexpr,
    parameter_lane: tl.constexpr,
    bit: tl.constexpr,
    block_pairs: tl.constexpr,
):
    lane = tl.program_id(1)
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = pairs < dimension // 2
    index0 = _insert_zero_bit(pairs, bit)
    index1 = index0 | (1 << bit)
    offset = lane * dimension
    r0 = tl.load(augmented_in + 2 * (offset + index0), mask=mask, other=0.0)
    i0 = tl.load(augmented_in + 2 * (offset + index0) + 1, mask=mask, other=0.0)
    r1 = tl.load(augmented_in + 2 * (offset + index1), mask=mask, other=0.0)
    i1 = tl.load(augmented_in + 2 * (offset + index1) + 1, mask=mask, other=0.0)
    half = 0.5 * tl.load(angle_ptr)
    sine, cosine = tl.sin(half), tl.cos(half)
    nr0, ni0 = cosine * r0 + sine * i0, cosine * i0 - sine * r0
    nr1, ni1 = cosine * r1 - sine * i1, cosine * i1 + sine * r1
    selected = lane == parameter_lane
    state_r0 = tl.load(augmented_in + 2 * index0, mask=mask, other=0.0)
    state_i0 = tl.load(augmented_in + 2 * index0 + 1, mask=mask, other=0.0)
    state_r1 = tl.load(augmented_in + 2 * index1, mask=mask, other=0.0)
    state_i1 = tl.load(augmented_in + 2 * index1 + 1, mask=mask, other=0.0)
    nr0 += tl.where(selected, 0.5 * (-sine * state_r0 + cosine * state_i0), 0.0)
    ni0 += tl.where(selected, 0.5 * (-sine * state_i0 - cosine * state_r0), 0.0)
    nr1 += tl.where(selected, 0.5 * (-sine * state_r1 - cosine * state_i1), 0.0)
    ni1 += tl.where(selected, 0.5 * (-sine * state_i1 + cosine * state_r1), 0.0)
    tl.store(augmented_out + 2 * (offset + index0), nr0, mask=mask)
    tl.store(augmented_out + 2 * (offset + index0) + 1, ni0, mask=mask)
    tl.store(augmented_out + 2 * (offset + index1), nr1, mask=mask)
    tl.store(augmented_out + 2 * (offset + index1) + 1, ni1, mask=mask)


def heisenberg_hva_forward_tangents(
    initial_state: torch.Tensor,
    parameters: torch.Tensor,
    *,
    n_wires: int,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the complete bond-resolved-phase HVA with forward tangents."""

    count = (3 * (n_wires - 1) + n_wires) * depth
    if not initial_state.is_cuda or initial_state.dtype != torch.complex64:
        raise ValueError("the experimental HVA tangent runtime requires CUDA complex64")
    if initial_state.numel() != 1 << n_wires or parameters.shape != (count,):
        raise ValueError("initial state or parameter shape does not match HVA")
    dimension, lanes = 1 << n_wires, count + 1
    augmented = torch.zeros(
        lanes, dimension, dtype=torch.complex64, device=initial_state.device
    )
    augmented[0] = initial_state.reshape(-1)
    scratch = torch.empty_like(augmented)
    block_groups, block_pairs = 128, 256
    per_layer = 3 * (n_wires - 1) + n_wires
    for layer in range(depth):
        offset = layer * per_layer
        for family in range(3):
            for left in range(n_wires - 1):
                parameter = offset + family * (n_wires - 1) + left
                grid = (triton.cdiv(dimension // 4, block_groups), lanes)
                _two_qubit_forward_tangent_kernel[grid](
                    torch.view_as_real(augmented),
                    torch.view_as_real(scratch),
                    parameters[parameter : parameter + 1],
                    dimension=dimension,
                    lanes=lanes,
                    parameter_lane=parameter + 1,
                    bit_left=n_wires - 1 - left,
                    bit_right=n_wires - 2 - left,
                    family=family,
                    block_groups=block_groups,
                    num_warps=4,
                    num_stages=2,
                )
                augmented, scratch = scratch, augmented
        for wire in range(n_wires):
            parameter = offset + 3 * (n_wires - 1) + wire
            grid = (triton.cdiv(dimension // 2, block_pairs), lanes)
            _rz_forward_tangent_kernel[grid](
                torch.view_as_real(augmented),
                torch.view_as_real(scratch),
                parameters[parameter : parameter + 1],
                dimension=dimension,
                lanes=lanes,
                parameter_lane=parameter + 1,
                bit=n_wires - 1 - wire,
                block_pairs=block_pairs,
                num_warps=8,
                num_stages=2,
            )
            augmented, scratch = scratch, augmented
    return augmented[0], augmented[1:]


__all__ = ["heisenberg_hva_forward_tangents"]
