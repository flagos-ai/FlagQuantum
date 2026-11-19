"""Apply the double-slit block independently as indexed wave-tensor transitions."""

import math

import torch

from .model import COMPLEX, REAL, Inverse, Model

H = torch.tensor(((1, 1), (1, -1)), dtype=COMPLEX) / math.sqrt(2)
COORDINATES = torch.arange(16)
X_BOUNDARY = ((COORDINATES == 0) | (COORDINATES == 15))[:, None]
Y_BOUNDARY = X_BOUNDARY.T
BARRIER = (~torch.isin(COORDINATES // 2, torch.tensor((2, 5))))[:, None] & (
    (COORDINATES // 2) == 3
)[None, :]
EDGE_MASK = X_BOUNDARY | Y_BOUNDARY | BARRIER


def axis_h(state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,ajdsexy->aidsexy", H, state)


def shift_h(state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,abjsexy->abisexy", H, axis_h(state))


def laplacian(state: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
    if adjoint:
        state = torch.where(EDGE_MASK, state.flip((4,)), state)
    else:
        state = torch.where(BARRIER, state.flip((3,)), state)
    state = shift_h(state)
    shifted = torch.empty_like(state)
    for axis in (0, 1):
        for choice in (0, 1):
            direction = -1 if bool(choice) != adjoint else 1
            shifted[:, axis, choice] = state[:, axis, choice].roll(
                direction, dims=-2 + axis
            )
    state = shift_h(shifted)
    if adjoint:
        return torch.where(BARRIER, state.flip((3,)), state)
    return torch.where(EDGE_MASK, state.flip((4,)), state)


def absorbing_boundary(state: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
    non_corner = ~(X_BOUNDARY ^ Y_BOUNDARY)
    if adjoint:
        state = torch.where(non_corner, state.flip((3,)), state)
    state = axis_h(state)
    updated = torch.empty_like(state)
    for axis, boundary in enumerate((X_BOUNDARY, Y_BOUNDARY)):
        selected = state[:, axis].clone()
        coordinates = COORDINATES ^ 1
        if adjoint:
            selected = torch.where(~boundary, selected.flip((1,)), selected)
            selected = selected.index_select(selected.ndim - 2 + axis, coordinates)
        else:
            selected = selected.index_select(selected.ndim - 2 + axis, coordinates)
            selected = torch.where(~boundary, selected.flip((1,)), selected)
        updated[:, axis] = selected
    updated = axis_h(updated)
    if adjoint:
        return updated
    return torch.where(non_corner, updated.flip((3,)), updated)


def real_rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(((c, -s), (s, c)), dtype=COMPLEX)


def selector_reference(weights: tuple[float, ...]) -> torch.Tensor:
    first, second, third, fourth = weights
    low = real_rotation(math.atan2(second, first))
    high = real_rotation(math.atan2(fourth, third))
    select = real_rotation(
        math.atan2(math.hypot(third, fourth), math.hypot(first, second))
    )
    controlled = torch.zeros((4, 4), dtype=COMPLEX)
    controlled[:2, :2], controlled[2:, 2:] = low, high
    mixed = torch.kron(select, torch.eye(2, dtype=COMPLEX))
    matrix: torch.Tensor = controlled @ mixed
    return matrix


def apply_signal(
    state: torch.Tensor,
    selector: torch.Tensor,
    absorbing_phase: float,
    *,
    adjoint: bool,
) -> torch.Tensor:
    prepared = torch.einsum("ij,ajbdsexy->aibdsexy", selector, state)
    prepared[:, 0] = laplacian(prepared[:, 0], adjoint=adjoint)
    factor = torch.exp(
        torch.tensor((-1j if adjoint else 1j) * absorbing_phase, dtype=COMPLEX)
    )
    prepared[:, 1] = factor * absorbing_boundary(prepared[:, 1], adjoint=adjoint)
    prepared[:, 2] *= -1
    return torch.einsum("ij,ajbdsexy->aibdsexy", selector.mH, prepared)


def complete_state(model: Model, inverse: Inverse) -> torch.Tensor:
    select = selector_reference(model.selector_amplitudes)
    angles = inverse.phases.phases - math.pi / 2
    angles[0] += math.pi / 4
    angles[-1] += math.pi / 4
    state = torch.zeros((2, 4, 2, 2, 2, 2, 16, 16), dtype=COMPLEX)
    state[:, 0, 0, 0, 0, 0] = model.source.reshape(16, 16) / math.sqrt(2)
    phases = torch.full((4, 2, 2, 2, 2), -1, dtype=REAL)
    phases[0, 0, 0, 0, 0] = 1
    signs = torch.stack((phases, -phases))
    state *= (1j * angles[0] * signs[..., None, None]).exp()
    for index, angle in enumerate(angles[1:]):
        state = apply_signal(state, select, model.absorb_phase, adjoint=index % 2 == 0)
        state *= (1j * angle * signs[..., None, None]).exp()
    return (H @ state.reshape(2, -1)).flatten()
