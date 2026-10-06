"""Rank-local numerical scans for MPS observables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..matrices import GATE_MAT_DICT
from .observable_adjoint_dispatch import (
    _try_apply_cataloged_mps_observable_adjoint,
)
from .site_kernels import (
    _record_mps_observable_adjoint_fallback,
    _record_mps_observable_adjoint_route,
    environment_transfer,
    environment_transfer_channels,
)


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
    """Move a batched MPS operator environment from right to left.

    The four-operand contraction
    ``sum_{p,q,r,s} conj(T[i,p,r]) O[p,q] T[j,q,s] E[r,s]`` is a reassociation
    of two batched matrix products once the operator is folded into the bra
    tensor's physical leg, ``L[i,q,r] = sum_p conj(T[i,p,r]) O[p,q]``: the
    environment contracts against ``L`` over ``r``, and the result contracts
    against the ket tensor over the fused ``(q, s)`` pair. Naming that order
    matters because the operator is diagonal at every site but the measured one,
    and leaving the order to the einsum planner costs 2.8x at the recorded hot
    shape -- environment ``(1, 64, 64)``, tensor ``(1, 64, 2, 64)``, operator
    ``(2, 2)`` -- measured as a chain, which is what the scan is.
    """

    batch, bond, physical = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    lifted = torch.einsum("bipr,pq->biqr", tensor.conj(), operator)
    contracted = torch.bmm(
        lifted.reshape(batch, bond * physical, bond), environment
    ).reshape(batch, bond, physical * bond)
    ket = tensor.reshape(batch, bond, physical * bond)
    return torch.bmm(contracted, ket.transpose(1, 2))


def mps_local_observable_adjoint(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
    *,
    hermitian: bool = False,
) -> torch.Tensor:
    """Differentiate one local tensor contribution to a weighted observable.

    ``hermitian=True`` records that the caller has established the MPS-006
    semantic invariant for both environments and the local operator.
    """

    cataloged = _try_apply_cataloged_mps_observable_adjoint(
        tensor,
        left_environment,
        right_environment,
        operator,
        weights,
        hermitian=hermitian,
    )
    if cataloged is not None:
        _record_mps_observable_adjoint_route()
        return cataloged
    _record_mps_observable_adjoint_fallback()

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
    qubits: Sequence[int],
    z_terms: Mapping[int, Sequence[int]],
    zz_terms: Mapping[int, Sequence[int]],
    *,
    compiled: bool,
) -> tuple[torch.Tensor, ...]:
    """Advance all Z and adjacent-ZZ channels across one rank-local site block."""

    if len(tensors) != len(qubits):
        raise ValueError("MPS observable tensors and qubits must have equal length")
    norm, previous_z, *channel_values = inputs
    channels = torch.stack(channel_values)
    for qubit, tensor in zip(qubits, tensors, strict=True):
        next_norm = environment_transfer(norm, tensor, z=False, compiled=compiled)
        next_z = environment_transfer(norm, tensor, z=True, compiled=compiled)
        channel_list = list(
            environment_transfer_channels(
                channels,
                tensor,
                compiled=compiled,
            ).unbind(0)
        )
        for index in z_terms.get(int(qubit), ()):
            channel_list[index] = next_z
        for index in zz_terms.get(int(qubit), ()):
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
    qubits: Sequence[int],
    coefficients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Advance the five-channel nearest-neighbor Heisenberg MPO on one rank."""

    if len(inputs) != 5:
        raise ValueError("Heisenberg MPS scan requires five input channels")
    if len(tensors) != len(qubits):
        raise ValueError("Heisenberg MPS tensors and qubits must have equal length")
    field_z, coupling_x, coupling_y, coupling_z = coefficients
    norm, open_x, open_y, open_z, energy = inputs
    operator_cache: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = {}
    for qubit, tensor in zip(qubits, tensors, strict=True):
        key = (tensor.device, tensor.dtype)
        operators = operator_cache.get(key)
        if operators is None:
            operators = {}
            for name in ("i", "x", "y", "z"):
                matrix = GATE_MAT_DICT[name]
                if not isinstance(matrix, torch.Tensor):
                    raise ValueError("Heisenberg scans require fixed Pauli matrices.")
                operators[name] = matrix.to(tensor.device, tensor.dtype)
            operator_cache[key] = operators
        next_norm = transfer_mps_operator_environment(norm, tensor, operators["i"])
        next_x = transfer_mps_operator_environment(norm, tensor, operators["x"])
        next_y = transfer_mps_operator_environment(norm, tensor, operators["y"])
        next_z = transfer_mps_operator_environment(norm, tensor, operators["z"])
        next_energy = transfer_mps_operator_environment(
            energy,
            tensor,
            operators["i"],
        )
        next_energy = next_energy + field_z[qubit] * next_z
        if qubit > 0:
            for coupling, opened, axis in (
                (coupling_x, open_x, "x"),
                (coupling_y, open_y, "y"),
                (coupling_z, open_z, "z"),
            ):
                next_energy = next_energy + coupling[
                    qubit - 1
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
