"""Exact finite Ising actions for the approved four-sigma vacuum profile.

This internal Simulation entry consumes the existing Core source. It owns no
planner, elapsed time, Hamiltonian, integrator or execution-result wrapper.
"""

from __future__ import annotations

import cmath
import math

import torch

from ..core.ir import IRValidationError
from ..core.ir.program.model import Operation, ProgramRecord
from ..core.ir.program.physical import (
    IsingContext,
    IsingExchange,
    check_ising_exchange,
)
from ..core.ir.program.verifier import verify_program_record
from ..errors import CapabilityError, ExecutionError, ValidationError
from ..operators import SuperOperator
from .lindblad_request import _require_density_matrices


def _fusion_matrices(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh F/R matrices in the reviewed Nayak Ising gauge."""
    fusion = torch.tensor(
        [[1, 1], [1, -1]], dtype=torch.complex128, device=device
    ) / math.sqrt(2)
    exchange = torch.diag(
        torch.tensor(
            [cmath.exp(-1j * math.pi / 8), cmath.exp(3j * math.pi / 8)],
            dtype=torch.complex128,
            device=device,
        )
    )
    return fusion, exchange


def _exchange_unitary(
    exchange: IsingExchange, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Read one adjacent generator, retaining its complex global phase."""
    check_ising_exchange(exchange)
    fusion, outer = _fusion_matrices(device)
    forward = fusion.mH @ outer @ fusion if exchange.left == 1 else outer
    return forward.mH if exchange.clockwise else forward


def _ising_actions(record: ProgramRecord) -> tuple[IsingContext, tuple[Operation, ...]]:
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise IRValidationError(
            "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
        )
    context = record.physical_context
    if not isinstance(context, IsingContext):
        raise CapabilityError("finite Ising actions require the native Ising profile")
    return context, record.functions[0].blocks[0].operations


def _initial_density(
    context: IsingContext, device: torch.device | str = "cpu"
) -> torch.Tensor:
    state = torch.zeros((2, 2), dtype=torch.complex128, device=device)
    initial = 0 if context.initial_channel == "1" else 1
    state[initial, initial] = 1
    return state


def _charge_projectors(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    return torch.diag_embed(torch.eye(2, dtype=torch.complex128, device=device)).unbind(
        0
    )


def _apply_ising_actions(
    state: torch.Tensor, operations: tuple[Operation, ...]
) -> torch.Tensor:
    for operation in operations:
        if operation.opcode == "ising.dephase_pair":
            state = SuperOperator.from_kraus(_charge_projectors(state.device)).apply(
                state
            )
            continue
        if operation.opcode == "ising.measure_pair":
            raise CapabilityError("sampled Ising actions require a selected outcome")
        exchange = operation.physical_data
        assert isinstance(exchange, IsingExchange)
        unitary = _exchange_unitary(exchange, state.device)
        state = SuperOperator.left_right_multiply(unitary, unitary.mH).apply(state)
    return state


def _sampled_boundary(operations: tuple[Operation, ...]) -> int:
    for index, operation in enumerate(operations):
        if operation.opcode == "ising.measure_pair":
            return index
    raise CapabilityError("conditional Ising actions require a sampled observation")


def _ising_sampled_prefix(
    record: ProgramRecord, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Evolve to the sampled boundary without drawing or modifying any RNG."""
    context, operations = _ising_actions(record)
    boundary = _sampled_boundary(operations)
    return _apply_ising_actions(
        _initial_density(context, device), operations[:boundary]
    )


def _continue_ising_sampled(
    record: ProgramRecord, prefix_density: torch.Tensor, selected_index: int
) -> torch.Tensor:
    """Apply a Runtime-selected branch and its ordered continuation.

    Runtime owns the probability check, private draw and measurement record.
    This internal numerical seam rejects invalid or zero-weight selections and
    normalizes by the original branch trace, never a corrected sampling weight.
    """
    _, operations = _ising_actions(record)
    boundary = _sampled_boundary(operations)
    if type(selected_index) is not int or selected_index not in (0, 1):
        raise ExecutionError("selected Ising outcome must be index 0 or 1")
    if (
        prefix_density.shape != (2, 2)
        or prefix_density.dtype != torch.complex128
        or prefix_density.device.type not in {"cpu", "flagos"}
        or prefix_density.requires_grad
        or not bool(torch.isfinite(prefix_density).all())
    ):
        raise ExecutionError(
            "Ising prefix must be a finite CPU complex128 (2, 2) density"
        )
    try:
        _require_density_matrices(prefix_density, tolerance=1e-10)
    except ValidationError as error:
        raise ExecutionError("invalid Ising prefix density") from error
    branch = SuperOperator.from_kraus(
        (_charge_projectors(prefix_density.device)[selected_index],)
    ).apply(prefix_density)
    probability = torch.trace(branch).real
    if not bool(torch.isfinite(probability)) or probability <= 0:
        raise ExecutionError("selected Ising branch must have a positive finite trace")
    return _apply_ising_actions(branch / probability, operations[boundary + 1 :])


def evolve_ising_braid(
    record: ProgramRecord, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Apply ordered braids and unread charge actions to the initial density.

    Return one complex128 (2, 2) density matrix on the requested device in canonical fusion order.
    Runtime owns input admission, batching, planning, outputs and result evidence;
    this entry does not activate native execution through fq.plan or fq.run.
    """
    context, operations = _ising_actions(record)
    if any(operation.opcode == "ising.measure_pair" for operation in operations):
        raise CapabilityError(
            "sampled Ising actions require a Runtime-selected outcome"
        )
    return _apply_ising_actions(_initial_density(context, device), operations)
