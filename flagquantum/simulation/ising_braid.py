"""Exact finite Ising actions for the approved four-sigma vacuum profile.

This internal Simulation entry consumes the existing Core source. It owns no
planner, elapsed time, Hamiltonian, integrator or execution-result wrapper.
"""

from __future__ import annotations

import cmath
import math

import torch

from ..core.ir import IRValidationError
from ..core.ir.program.model import ProgramRecord
from ..core.ir.program.physical import (
    IsingContext,
    IsingExchange,
    check_ising_exchange,
)
from ..core.ir.program.verifier import verify_program_record
from ..errors import CapabilityError
from ..operators import SuperOperator


def _fusion_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh F/R matrices in the reviewed Nayak Ising gauge."""
    fusion = torch.tensor(
        [[1, 1], [1, -1]], dtype=torch.complex128, device="cpu"
    ) / math.sqrt(2)
    exchange = torch.diag(
        torch.tensor(
            [cmath.exp(-1j * math.pi / 8), cmath.exp(3j * math.pi / 8)],
            dtype=torch.complex128,
            device="cpu",
        )
    )
    return fusion, exchange


def _exchange_unitary(exchange: IsingExchange) -> torch.Tensor:
    """Read one adjacent generator, retaining its complex global phase."""
    check_ising_exchange(exchange)
    fusion, outer = _fusion_matrices()
    forward = fusion.mH @ outer @ fusion if exchange.left == 1 else outer
    return forward.mH if exchange.clockwise else forward


def evolve_ising_braid(record: ProgramRecord) -> torch.Tensor:
    """Apply ordered braids and unread charge actions to the initial density.

    Return one CPU complex128 (2, 2) density matrix in canonical fusion order.
    Runtime owns input admission, batching, planning, outputs and result evidence;
    this entry does not activate native execution through fq.plan or fq.run.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise IRValidationError(
            "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
        )
    context = record.physical_context
    if not isinstance(context, IsingContext):
        raise CapabilityError("finite Ising actions require the native Ising profile")
    state = torch.zeros((2, 2), dtype=torch.complex128, device="cpu")
    initial = 0 if context.initial_channel == "1" else 1
    state[initial, initial] = 1
    for operation in record.functions[0].blocks[0].operations:
        if operation.opcode == "ising.dephase_pair":
            projectors = torch.diag_embed(
                torch.eye(2, dtype=state.dtype, device=state.device)
            ).unbind(0)
            state = SuperOperator.from_kraus(projectors).apply(state)
            continue
        exchange = operation.physical_data
        assert isinstance(exchange, IsingExchange)
        unitary = _exchange_unitary(exchange)
        state = SuperOperator.left_right_multiply(unitary, unitary.mH).apply(state)
    return state
