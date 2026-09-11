"""Persistent exact tangents for repeated two-qubit Pauli rotations."""

from __future__ import annotations

from typing import Callable, Protocol

import torch
import triton
import triton.language as tl


class _TensorJacobian(Protocol):
    def __call__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        *,
        vectorize: bool,
    ) -> object: ...


@triton.jit
def _rxx_ryy_rzz_tangent_kernel(
    state_parts: tl.tensor,
    angles: tl.tensor,
    tangent_parts: tl.tensor,
    group_count: tl.tensor,
    depth: tl.constexpr,
    state_batch_stride: tl.constexpr,
    angle_batch_stride: tl.constexpr,
    angle_layer_stride: tl.constexpr,
    tangent_parameter_stride: tl.constexpr,
    tangent_batch_stride: tl.constexpr,
    block_groups: tl.constexpr,
) -> None:
    parameter = tl.program_id(2)
    batch = tl.program_id(1)
    groups = tl.program_id(0) * block_groups + tl.arange(0, block_groups)
    mask = groups < group_count
    base = batch * state_batch_stride + groups * 4
    r0 = tl.load(state_parts + 2 * base, mask=mask, other=0.0)
    i0 = tl.load(state_parts + 2 * base + 1, mask=mask, other=0.0)
    r1 = tl.load(state_parts + 2 * (base + 1), mask=mask, other=0.0)
    i1 = tl.load(state_parts + 2 * (base + 1) + 1, mask=mask, other=0.0)
    r2 = tl.load(state_parts + 2 * (base + 2), mask=mask, other=0.0)
    i2 = tl.load(state_parts + 2 * (base + 2) + 1, mask=mask, other=0.0)
    r3 = tl.load(state_parts + 2 * (base + 3), mask=mask, other=0.0)
    i3 = tl.load(state_parts + 2 * (base + 3) + 1, mask=mask, other=0.0)
    tr0 = tl.zeros((block_groups,), tl.float32)
    ti0 = tl.zeros((block_groups,), tl.float32)
    tr1 = tl.zeros((block_groups,), tl.float32)
    ti1 = tl.zeros((block_groups,), tl.float32)
    tr2 = tl.zeros((block_groups,), tl.float32)
    ti2 = tl.zeros((block_groups,), tl.float32)
    tr3 = tl.zeros((block_groups,), tl.float32)
    ti3 = tl.zeros((block_groups,), tl.float32)
    target_layer = parameter // 3
    target_family = parameter % 3
    for layer in range(depth):
        for family in range(3):
            if family == 0:  # XX: 00<->11, 01<->10
                pr0, pi0, ptr0, pti0 = r3, i3, tr3, ti3
                pr1, pi1, ptr1, pti1 = r2, i2, tr2, ti2
                pr2, pi2, ptr2, pti2 = r1, i1, tr1, ti1
                pr3, pi3, ptr3, pti3 = r0, i0, tr0, ti0
            elif family == 1:  # YY: negative on 00<->11
                pr0, pi0, ptr0, pti0 = -r3, -i3, -tr3, -ti3
                pr1, pi1, ptr1, pti1 = r2, i2, tr2, ti2
                pr2, pi2, ptr2, pti2 = r1, i1, tr1, ti1
                pr3, pi3, ptr3, pti3 = -r0, -i0, -tr0, -ti0
            else:  # ZZ: diag(1,-1,-1,1)
                pr0, pi0, ptr0, pti0 = r0, i0, tr0, ti0
                pr1, pi1, ptr1, pti1 = -r1, -i1, -tr1, -ti1
                pr2, pi2, ptr2, pti2 = -r2, -i2, -tr2, -ti2
                pr3, pi3, ptr3, pti3 = r3, i3, tr3, ti3
            offset = batch * angle_batch_stride + layer * angle_layer_stride + family
            half_angle = 0.5 * tl.load(angles + offset)
            sine, cosine = tl.sin(half_angle), tl.cos(half_angle)
            nr0, ni0 = cosine * r0 + sine * pi0, cosine * i0 - sine * pr0
            nr1, ni1 = cosine * r1 + sine * pi1, cosine * i1 - sine * pr1
            nr2, ni2 = cosine * r2 + sine * pi2, cosine * i2 - sine * pr2
            nr3, ni3 = cosine * r3 + sine * pi3, cosine * i3 - sine * pr3
            ntr0, nti0 = cosine * tr0 + sine * pti0, cosine * ti0 - sine * ptr0
            ntr1, nti1 = cosine * tr1 + sine * pti1, cosine * ti1 - sine * ptr1
            ntr2, nti2 = cosine * tr2 + sine * pti2, cosine * ti2 - sine * ptr2
            ntr3, nti3 = cosine * tr3 + sine * pti3, cosine * ti3 - sine * ptr3
            selected = (layer == target_layer) & (family == target_family)
            ntr0 += tl.where(selected, 0.5 * (-sine * r0 + cosine * pi0), 0.0)
            nti0 += tl.where(selected, 0.5 * (-sine * i0 - cosine * pr0), 0.0)
            ntr1 += tl.where(selected, 0.5 * (-sine * r1 + cosine * pi1), 0.0)
            nti1 += tl.where(selected, 0.5 * (-sine * i1 - cosine * pr1), 0.0)
            ntr2 += tl.where(selected, 0.5 * (-sine * r2 + cosine * pi2), 0.0)
            nti2 += tl.where(selected, 0.5 * (-sine * i2 - cosine * pr2), 0.0)
            ntr3 += tl.where(selected, 0.5 * (-sine * r3 + cosine * pi3), 0.0)
            nti3 += tl.where(selected, 0.5 * (-sine * i3 - cosine * pr3), 0.0)
            r0, i0, tr0, ti0 = nr0, ni0, ntr0, nti0
            r1, i1, tr1, ti1 = nr1, ni1, ntr1, nti1
            r2, i2, tr2, ti2 = nr2, ni2, ntr2, nti2
            r3, i3, tr3, ti3 = nr3, ni3, ntr3, nti3
    tangent_base = (
        parameter * tangent_parameter_stride + batch * tangent_batch_stride + groups * 4
    )
    tl.store(tangent_parts + 2 * tangent_base, tr0, mask=mask)
    tl.store(tangent_parts + 2 * tangent_base + 1, ti0, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 1), tr1, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 1) + 1, ti1, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 2), tr2, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 2) + 1, ti2, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 3), tr3, mask=mask)
    tl.store(tangent_parts + 2 * (tangent_base + 3) + 1, ti3, mask=mask)


def _reference(state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    xx = torch.tensor(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        device=state.device,
        dtype=state.dtype,
    )
    yy = torch.tensor(
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
        device=state.device,
        dtype=state.dtype,
    )
    zz = torch.diag(
        torch.tensor([1, -1, -1, 1], device=state.device, dtype=state.dtype)
    )
    output = state
    for layer in range(int(angles.shape[1])):
        for family, pauli in enumerate((xx, yy, zz)):
            half = angles[:, layer, family].reshape(-1, 1, 1) / 2
            transformed = torch.einsum("ij,bgj->bgi", pauli, output)
            output = torch.cos(half) * output - 1j * torch.sin(half) * transformed
    return output


def repeated_rxx_ryy_rzz_tangents(
    state: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """Return layer-major RXX/RYY/RZZ tangents as ``[3*depth,batch,groups,4]``."""

    if state.ndim != 3 or int(state.shape[-1]) != 4:
        raise ValueError("state must have shape [batch, groups, 4]")
    if (
        angles.ndim != 3
        or int(angles.shape[0]) != int(state.shape[0])
        or int(angles.shape[2]) != 3
    ):
        raise ValueError("angles must have shape [batch, depth, 3]")
    depth = int(angles.shape[1])
    if not state.is_cuda or state.dtype != torch.complex64:
        differentiate: _TensorJacobian = torch.autograd.functional.jacobian
        components: list[torch.Tensor] = []
        for component in (
            lambda value: _reference(state, value).real,
            lambda value: _reference(state, value).imag,
        ):
            result = differentiate(component, angles, vectorize=True)
            if not isinstance(result, torch.Tensor):
                raise TypeError("Pauli rotation Jacobian must return a tensor")
            components.append(result)
        real, imag = components
        jacobian = torch.complex(real, imag)
        indices = torch.arange(int(state.shape[0]), device=state.device)
        return (
            jacobian[indices, :, :, indices]
            .movedim((-2, -1), (0, 1))
            .reshape(3 * depth, *state.shape)
        )
    state = state.contiguous()
    angles = angles.contiguous()
    batch, groups, _ = state.shape
    tangent_parts = torch.empty(
        3 * depth, batch, groups, 4, 2, device=state.device, dtype=torch.float32
    )
    block_groups = 128
    grid = (triton.cdiv(groups, block_groups), batch, 3 * depth)
    _rxx_ryy_rzz_tangent_kernel[grid](
        torch.view_as_real(state),
        angles,
        tangent_parts,
        groups,
        depth=depth,
        state_batch_stride=state.stride(0),
        angle_batch_stride=angles.stride(0),
        angle_layer_stride=angles.stride(1),
        tangent_parameter_stride=tangent_parts.stride(0) // 2,
        tangent_batch_stride=tangent_parts.stride(1) // 2,
        block_groups=block_groups,
        num_warps=4,
        num_stages=2,
    )
    return torch.view_as_complex(tangent_parts)


__all__ = ["repeated_rxx_ryy_rzz_tangents"]
