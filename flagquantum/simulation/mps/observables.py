"""Rank-local numerical scans for MPS observables."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ...ops.matrices import GATE_MAT_DICT
from .site_kernels import environment_transfer, environment_transfer_channels


def transfer_mps_operator_environment(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Advance a batched MPS environment through one explicit operator."""

    return torch.einsum(
        "bij,bipr,pq,bjqs->brs",
        environment,
        tensor.conj(),
        operator,
        tensor,
    )


def transfer_mps_operator_right_environment(
    tensor: torch.Tensor,
    operator: torch.Tensor,
    environment: torch.Tensor,
) -> torch.Tensor:
    """Move a batched MPS operator environment from right to left."""

    return torch.einsum(
        "bipr,pq,bjqs,brs->bij",
        tensor.conj(),
        operator,
        tensor,
        environment,
    )


def mps_local_observable_adjoint(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiate one local tensor contribution to a weighted observable."""

    variable = tensor.detach().requires_grad_(True)
    local_values = torch.real(
        torch.einsum(
            "bij,bipr,pq,bjqs,brs->b",
            left_environment,
            variable.conj(),
            operator,
            variable,
            right_environment,
        )
    )
    return torch.autograd.grad(torch.sum(weights * local_values), variable)[0].detach()


def mps_z_zz_local_scan(
    inputs: Sequence[torch.Tensor],
    tensors: Sequence[torch.Tensor],
    wires: Sequence[int],
    z_terms: Mapping[int, Sequence[int]],
    zz_terms: Mapping[int, Sequence[int]],
    *,
    compiled: bool,
) -> tuple[torch.Tensor, ...]:
    """Advance all Z and adjacent-ZZ channels across one rank-local site block."""

    if len(tensors) != len(wires):
        raise ValueError("MPS observable tensors and wires must have equal length")
    norm, previous_z, *channel_values = inputs
    channels = torch.stack(channel_values)
    for wire, tensor in zip(wires, tensors):
        next_norm = environment_transfer(norm, tensor, z=False, compiled=compiled)
        next_z = environment_transfer(norm, tensor, z=True, compiled=compiled)
        channel_list = list(
            environment_transfer_channels(
                channels,
                tensor,
                compiled=compiled,
            ).unbind(0)
        )
        for index in z_terms.get(int(wire), ()):
            channel_list[index] = next_z
        for index in zz_terms.get(int(wire), ()):
            channel_list[index] = environment_transfer(
                previous_z,
                tensor,
                z=True,
                compiled=compiled,
            )
        norm, previous_z, channels = next_norm, next_z, torch.stack(channel_list)
    return (norm, previous_z) + tuple(channels.unbind(0))


def mps_heisenberg_local_scan(
    inputs: Sequence[torch.Tensor],
    tensors: Sequence[torch.Tensor],
    wires: Sequence[int],
    coefficients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Advance the five-channel nearest-neighbor Heisenberg MPO on one rank."""

    if len(inputs) != 5:
        raise ValueError("Heisenberg MPS scan requires five input channels")
    if len(tensors) != len(wires):
        raise ValueError("Heisenberg MPS tensors and wires must have equal length")
    field_z, coupling_x, coupling_y, coupling_z = coefficients
    norm, open_x, open_y, open_z, energy = inputs
    for wire, tensor in zip(wires, tensors):
        operators = {
            name: GATE_MAT_DICT[name].to(tensor.device, tensor.dtype)
            for name in ("i", "x", "y", "z")
        }
        next_norm = transfer_mps_operator_environment(norm, tensor, operators["i"])
        next_x = transfer_mps_operator_environment(norm, tensor, operators["x"])
        next_y = transfer_mps_operator_environment(norm, tensor, operators["y"])
        next_z = transfer_mps_operator_environment(norm, tensor, operators["z"])
        next_energy = transfer_mps_operator_environment(
            energy,
            tensor,
            operators["i"],
        )
        next_energy = next_energy + field_z[wire] * next_z
        if wire > 0:
            for coupling, opened, axis in (
                (coupling_x, open_x, "x"),
                (coupling_y, open_y, "y"),
                (coupling_z, open_z, "z"),
            ):
                next_energy = next_energy + coupling[
                    wire - 1
                ] * transfer_mps_operator_environment(
                    opened,
                    tensor,
                    operators[axis],
                )
        norm, open_x, open_y, open_z, energy = (
            next_norm,
            next_x,
            next_y,
            next_z,
            next_energy,
        )
    return norm, open_x, open_y, open_z, energy


__all__ = (
    "mps_heisenberg_local_scan",
    "mps_local_observable_adjoint",
    "mps_z_zz_local_scan",
    "transfer_mps_operator_environment",
    "transfer_mps_operator_right_environment",
)
