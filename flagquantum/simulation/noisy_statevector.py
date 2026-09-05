"""Numerical kernels for batched statevector quantum trajectories."""

from __future__ import annotations

import torch


def apply_matrix_batched(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    n_wires: int,
) -> torch.Tensor:
    """Apply an unbatched or circuit-batched matrix to trajectory states."""

    trajectories, circuit_batch, _ = state.shape
    width = len(wires)
    dimension = 2**width
    other_wires = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = (0, 1) + tuple(wire + 2 for wire in wires + other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    flat = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories * circuit_batch, dimension, -1)
    )
    matrix = torch.as_tensor(matrix, device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        applied = torch.matmul(matrix, flat)
    elif matrix.ndim == 3 and matrix.shape[0] == circuit_batch:
        expanded = matrix.unsqueeze(0).expand(trajectories, -1, -1, -1)
        applied = torch.bmm(expanded.reshape(-1, dimension, dimension), flat)
    else:
        raise ValueError(
            "gate matrix must be unbatched or match the circuit batch dimension"
        )
    return (
        applied.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape(trajectories, circuit_batch, -1)
    )


def sample_rows(
    probabilities: torch.Tensor,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample one categorical choice per trajectory row."""

    choices = [
        torch.multinomial(row, 1, replacement=True, generator=generator).squeeze(-1)
        for row, generator in zip(probabilities, generators, strict=True)
    ]
    return torch.stack(choices)


def apply_kraus_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wires: tuple[int, ...],
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample and normalize a generic Kraus branch for each trajectory."""

    branches = torch.stack(
        [
            apply_matrix_batched(state, operator, wires, n_wires)
            for operator in operators
        ],
        dim=2,
    )
    probabilities = torch.sum(torch.abs(branches) ** 2, dim=-1).real
    probabilities = torch.clamp(probabilities, min=0)
    totals = probabilities.sum(dim=-1, keepdim=True)
    if bool(torch.any(totals <= 0)):
        raise RuntimeError("Kraus channel produced zero total branch probability")
    choices = sample_rows(probabilities / totals, generators)
    selected = torch.gather(
        branches,
        2,
        choices[..., None, None].expand(-1, -1, 1, branches.shape[-1]),
    ).squeeze(2)
    selected_probability = torch.gather(probabilities, 2, choices[..., None]).squeeze(
        -1
    )
    return (
        selected / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None]
    )


def apply_amplitude_damping_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wire: int,
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample amplitude damping without materializing all branch states."""

    gamma = torch.abs(operators[1][0, 1]) ** 2
    trajectories, circuit_batch, _ = state.shape
    other_wires = tuple(index for index in range(n_wires) if index != wire)
    permutation = (0, 1, wire + 2) + tuple(index + 2 for index in other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    packed = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories, circuit_batch, 2, -1)
    )
    zero, one = packed[:, :, 0], packed[:, :, 1]
    jump_probability = torch.clamp(
        gamma.real * torch.sum(torch.abs(one) ** 2, dim=-1), min=0, max=1
    )
    probabilities = torch.stack((1 - jump_probability, jump_probability), dim=-1)
    choices = sample_rows(probabilities, generators)
    jump = choices.bool()[..., None]
    out_zero = torch.where(jump, torch.sqrt(gamma) * one, zero)
    out_one = torch.where(jump, torch.zeros_like(one), torch.sqrt(1 - gamma) * one)
    selected_probability = torch.where(
        jump[..., 0], jump_probability, 1 - jump_probability
    )
    packed_out = (
        torch.stack((out_zero, out_one), dim=2)
        / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None, None]
    )
    return (
        packed_out.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape_as(state)
    )


def expectation_z(state: torch.Tensor, n_wires: int) -> torch.Tensor:
    """Return per-trajectory Z expectations for every wire."""

    indices = torch.arange(state.shape[-1], device=state.device)
    shifts = torch.arange(n_wires - 1, -1, -1, device=state.device)
    signs = 1 - 2 * ((indices[:, None] >> shifts) & 1)
    return torch.matmul(torch.abs(state) ** 2, signs.to(dtype=state.real.dtype))


__all__ = (
    "apply_amplitude_damping_batched",
    "apply_kraus_batched",
    "apply_matrix_batched",
    "expectation_z",
    "sample_rows",
)
