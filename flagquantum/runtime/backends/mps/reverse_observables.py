"""Observable contracts and parsers for distributed MPS reverse execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....ops.matrices import GATE_MAT_DICT
from .reverse_transport import (
    receive_reverse_tensor,
    receive_static_reverse_tensor,
    send_reverse_tensor,
    send_static_reverse_tensor,
)
from .state import RankOwnedMPSState

_recv = receive_reverse_tensor
_send = send_reverse_tensor


@dataclass(frozen=True)
class SiteShardedZZScanResult:
    """Reusable value/objective result from one site-sharded Z/ZZ scan."""

    values: torch.Tensor
    loss: torch.Tensor | None
    adjoints: Mapping[int, torch.Tensor]
    forward_scans: int
    reverse_scans: int
    forward_messages: int
    reverse_messages: int


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


def mps_expectation_and_adjoints(
    state: RankOwnedMPSState,
    observable: Mapping[int, str],
    target: torch.Tensor | float | None = None,
    *,
    adjoint_wires: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Evaluate one Pauli observable and its rank-local tensor adjoints."""

    reference = next(iter(state.local_tensors.values()))
    requested_adjoint_wires = (
        set(state.local_tensors)
        if adjoint_wires is None
        else {int(wire) for wire in adjoint_wires}
    )
    local_adjoint_wires = requested_adjoint_wires.intersection(state.local_tensors)
    left_envs: dict[int, torch.Tensor] = {}
    env: torch.Tensor | None = None
    with record_function("flagquantum::mps::left_environment_scan"):
        for owner in range(state.world_size):
            if state.rank == owner:
                if owner == 0:
                    env = torch.ones(
                        state.bsz,
                        1,
                        1,
                        dtype=reference.dtype,
                        device=reference.device,
                    )
                assert env is not None
                for wire in state.ownership[owner]:
                    if wire in local_adjoint_wires:
                        left_envs[wire] = env
                    tensor = state.local_tensors[wire]
                    operator = GATE_MAT_DICT[str(observable.get(wire, "i")).lower()].to(
                        device=tensor.device, dtype=tensor.dtype
                    )
                    env = torch.einsum(
                        "bij,bipr,pq,bjqs->brs",
                        env,
                        tensor.conj(),
                        operator,
                        tensor,
                    )
            if owner < state.world_size - 1:
                sequence = 2_000_000 + owner
                if state.rank == owner:
                    assert env is not None
                    send_reverse_tensor(
                        env.unsqueeze(-1),
                        destination=owner + 1,
                        sequence=sequence,
                    )
                elif state.rank == owner + 1:
                    env = receive_reverse_tensor(
                        reference, source=owner, sequence=sequence
                    ).squeeze(-1)
    if state.rank == state.world_size - 1:
        assert env is not None
        expectation = torch.real(env[:, 0, 0]).detach().contiguous()
    else:
        expectation = torch.zeros(
            state.bsz, dtype=reference.real.dtype, device=reference.device
        )
    dist.broadcast(expectation, src=state.world_size - 1)
    if target is None:
        value = expectation.mean()
        weights = torch.full_like(expectation, 1.0 / state.bsz)
    else:
        resolved_target = torch.as_tensor(
            target, dtype=expectation.dtype, device=expectation.device
        ).reshape(-1)
        if resolved_target.numel() == 1:
            resolved_target = resolved_target.expand_as(expectation)
        if resolved_target.shape != expectation.shape:
            raise ValueError(
                "observable target must be scalar or have one value per MPS batch item"
            )
        residual = expectation - resolved_target
        value = residual.square().mean()
        weights = 2.0 * residual / state.bsz

    right_envs: dict[int, torch.Tensor] = {}
    right: torch.Tensor | None = None
    with record_function("flagquantum::mps::right_environment_scan"):
        for owner in range(state.world_size - 1, -1, -1):
            if state.rank == owner:
                if owner == state.world_size - 1:
                    right = torch.ones(
                        state.bsz,
                        1,
                        1,
                        dtype=reference.dtype,
                        device=reference.device,
                    )
                assert right is not None
                for wire in reversed(state.ownership[owner]):
                    if wire in local_adjoint_wires:
                        right_envs[wire] = right
                    tensor = state.local_tensors[wire]
                    operator = GATE_MAT_DICT[str(observable.get(wire, "i")).lower()].to(
                        device=tensor.device, dtype=tensor.dtype
                    )
                    right = torch.einsum(
                        "bipr,pq,bjqs,brs->bij",
                        tensor.conj(),
                        operator,
                        tensor,
                        right,
                    )
            if owner > 0:
                sequence = 2_100_000 + owner
                if state.rank == owner:
                    assert right is not None
                    send_reverse_tensor(
                        right.unsqueeze(-1),
                        destination=owner - 1,
                        sequence=sequence,
                    )
                elif state.rank == owner - 1:
                    right = receive_reverse_tensor(
                        reference, source=owner, sequence=sequence
                    ).squeeze(-1)

    adjoints = {}
    with record_function("flagquantum::mps::objective_adjoint"):
        for wire in sorted(local_adjoint_wires):
            tensor = state.local_tensors[wire]
            variable = tensor.detach().requires_grad_(True)
            operator = GATE_MAT_DICT[str(observable.get(wire, "i")).lower()].to(
                device=tensor.device, dtype=tensor.dtype
            )
            local_values = torch.real(
                torch.einsum(
                    "bij,bipr,pq,bjqs,brs->b",
                    left_envs[wire],
                    variable.conj(),
                    operator,
                    variable,
                    right_envs[wire],
                )
            )
            scalar = torch.sum(weights * local_values)
            adjoints[wire] = torch.autograd.grad(scalar, variable)[0].detach()
    return value, adjoints


def mps_heisenberg_energy_and_adjoints(
    state: RankOwnedMPSState,
    coefficients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Contract a nearest-neighbor Heisenberg MPO with five live channels."""

    field_z, coupling_x, coupling_y, coupling_z = coefficients
    reference = next(iter(state.local_tensors.values()))
    carry = None
    local_inputs = local_outputs = local_variables = None
    with record_function("flagquantum::mps::heisenberg_mpo_forward"):
        for owner in range(state.world_size):
            if state.rank == owner:
                if owner == 0:
                    identity = torch.ones(
                        state.bsz,
                        1,
                        1,
                        dtype=reference.dtype,
                        device=reference.device,
                    )
                    inputs = (identity,) + tuple(
                        torch.zeros_like(identity) for _ in range(4)
                    )
                else:
                    assert carry is not None
                    inputs = tuple(item.detach() for item in carry.unbind(0))
                inputs = tuple(item.requires_grad_(True) for item in inputs)
                variables = tuple(
                    state.local_tensors[wire].detach().requires_grad_(True)
                    for wire in state.ownership[owner]
                )
                norm, open_x, open_y, open_z, energy = inputs
                for wire, tensor in zip(state.ownership[owner], variables):
                    operators = {
                        name: GATE_MAT_DICT[name].to(tensor.device, tensor.dtype)
                        for name in ("i", "x", "y", "z")
                    }
                    next_norm = transfer_mps_operator_environment(
                        norm, tensor, operators["i"]
                    )
                    next_x = transfer_mps_operator_environment(
                        norm, tensor, operators["x"]
                    )
                    next_y = transfer_mps_operator_environment(
                        norm, tensor, operators["y"]
                    )
                    next_z = transfer_mps_operator_environment(
                        norm, tensor, operators["z"]
                    )
                    next_energy = transfer_mps_operator_environment(
                        energy, tensor, operators["i"]
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
                                opened, tensor, operators[axis]
                            )
                    norm, open_x, open_y, open_z, energy = (
                        next_norm,
                        next_x,
                        next_y,
                        next_z,
                        next_energy,
                    )
                outputs = (norm, open_x, open_y, open_z, energy)
                local_inputs, local_outputs, local_variables = (
                    inputs,
                    outputs,
                    variables,
                )
            if owner < state.world_size - 1:
                sequence = 2_700_000 + owner
                if state.rank == owner:
                    stacked_outputs = torch.stack([item.detach() for item in outputs])
                    send_static_reverse_tensor(
                        stacked_outputs,
                        destination=owner + 1,
                        sequence=sequence,
                        shape=stacked_outputs.shape,
                    )
                elif state.rank == owner + 1:
                    first_wire = state.ownership[state.rank][0]
                    left_dim = int(state.local_tensors[first_wire].shape[1])
                    carry = receive_static_reverse_tensor(
                        reference,
                        source=owner,
                        sequence=sequence,
                        shape=(5, state.bsz, left_dim, left_dim),
                    )

    assert (
        local_inputs is not None
        and local_outputs is not None
        and local_variables is not None
    )
    output_grads = None
    with record_function("flagquantum::mps::heisenberg_mpo_reverse"):
        for owner in range(state.world_size - 1, -1, -1):
            if state.rank == owner:
                if owner == state.world_size - 1:
                    objective = torch.real(local_outputs[-1][:, 0, 0]).mean()
                    derivatives = torch.autograd.grad(
                        objective,
                        local_inputs + local_variables,
                        allow_unused=True,
                    )
                    value = objective.detach()
                else:
                    assert output_grads is not None
                    derivatives = torch.autograd.grad(
                        local_outputs,
                        local_inputs + local_variables,
                        grad_outputs=output_grads,
                        allow_unused=True,
                    )
                input_grads = tuple(
                    torch.zeros_like(item) if gradient is None else gradient
                    for item, gradient in zip(
                        local_inputs, derivatives[: len(local_inputs)]
                    )
                )
                variable_grads = derivatives[len(local_inputs) :]
            if owner > 0:
                sequence = 2_800_000 + owner
                if state.rank == owner:
                    stacked_grads = torch.stack(input_grads)
                    send_static_reverse_tensor(
                        stacked_grads,
                        destination=owner - 1,
                        sequence=sequence,
                        shape=stacked_grads.shape,
                    )
                elif state.rank == owner - 1:
                    output_shape = (len(local_outputs), *local_outputs[0].shape)
                    received = receive_static_reverse_tensor(
                        reference,
                        source=owner,
                        sequence=sequence,
                        shape=output_shape,
                    )
                    output_grads = tuple(received.unbind(0))

    if state.rank != state.world_size - 1:
        value = torch.zeros((), dtype=reference.real.dtype, device=reference.device)
    dist.broadcast(value, src=state.world_size - 1)
    adjoints = {
        wire: torch.zeros_like(tensor) if gradient is None else gradient.detach()
        for wire, tensor, gradient in zip(
            state.ownership[state.rank],
            state.local_tensors.values(),
            variable_grads,
        )
    }
    return value, adjoints


def parse_mps_z_zz_terms(
    state: RankOwnedMPSState,
    terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]],
) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...] | None:
    """Normalize supported local Z and adjacent ZZ objective terms."""

    parsed = []
    reference = next(iter(state.local_tensors.values())).real
    for observable, target in terms:
        wires = tuple(
            sorted(
                int(wire)
                for wire, name in observable.items()
                if str(name).lower() != "i"
            )
        )
        if (
            not wires
            or len(wires) > 2
            or any(str(observable[wire]).lower() != "z" for wire in wires)
        ):
            return None
        if len(wires) == 2 and wires[1] != wires[0] + 1:
            return None
        value = torch.as_tensor(
            target, dtype=reference.dtype, device=reference.device
        ).reshape(-1)
        if value.numel() == 1:
            value = value.expand(state.bsz)
        if value.shape != (state.bsz,):
            raise ValueError(
                "Z/ZZ target must be scalar or have one value per batch item"
            )
        parsed.append((wires, value))
    return tuple(parsed)


def parse_mps_heisenberg_terms(
    state: RankOwnedMPSState,
    terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Normalize Z fields and adjacent XX/YY/ZZ Hamiltonian couplings."""

    if not terms:
        raise ValueError("hamiltonian_terms must contain at least one term")
    reference = next(iter(state.local_tensors.values())).real
    field_z = torch.zeros(state.n_wires, dtype=reference.dtype, device=reference.device)
    couplings = {
        axis: torch.zeros(
            max(0, state.n_wires - 1),
            dtype=reference.dtype,
            device=reference.device,
        )
        for axis in ("x", "y", "z")
    }
    for observable, coefficient in terms:
        ops = tuple(
            sorted(
                (int(wire), str(name).lower())
                for wire, name in observable.items()
                if str(name).lower() != "i"
            )
        )
        value = torch.as_tensor(
            coefficient, dtype=reference.dtype, device=reference.device
        )
        if value.numel() != 1:
            return None
        if len(ops) == 1 and ops[0][1] == "z":
            wire = ops[0][0]
            if wire < 0 or wire >= state.n_wires:
                return None
            field_z[wire] += value.reshape(())
            continue
        if (
            len(ops) == 2
            and ops[1][0] == ops[0][0] + 1
            and ops[0][1] == ops[1][1]
            and ops[0][1] in couplings
            and 0 <= ops[0][0] < state.n_wires - 1
        ):
            couplings[ops[0][1]][ops[0][0]] += value.reshape(())
            continue
        return None
    return field_z, couplings["x"], couplings["y"], couplings["z"]


__all__ = (
    "SiteShardedZZScanResult",
    "mps_expectation_and_adjoints",
    "mps_heisenberg_energy_and_adjoints",
    "parse_mps_heisenberg_terms",
    "parse_mps_z_zz_terms",
    "transfer_mps_operator_environment",
)
