"""Local noiseless MPS numerical execution."""

from __future__ import annotations

from typing import Any

import torch

from ..core.ir import CircuitIR
from ..ops.gate_matrix import gate_matrix
from .mps.models import CompiledMPSProgram
from .mps.state import MPSState


def _compose_one_qubit_matrices(
    new_matrix: torch.Tensor,
    current_matrix: torch.Tensor,
    *,
    bsz: int,
) -> torch.Tensor:
    if new_matrix.ndim == 2 and current_matrix.ndim == 2:
        return new_matrix @ current_matrix
    if new_matrix.ndim == 2:
        new_matrix = new_matrix.expand(int(bsz), -1, -1)
    if current_matrix.ndim == 2:
        current_matrix = current_matrix.expand(int(bsz), -1, -1)
    return torch.bmm(new_matrix, current_matrix)


def run_local_mps(
    ir: CircuitIR,
    mps: MPSState,
    *,
    parameter_bindings: tuple[torch.Tensor, ...] | None = None,
    program_cache: dict[tuple[Any, ...], Any] | None = None,
    fuse_single_qubit: bool = True,
    spatial_bucket: bool = True,
) -> MPSState:
    """Apply one validated IR to an initialized local MPS."""

    instructions = tuple(ir)
    signature = (
        mps.n_wires,
        mps.bsz,
        str(mps.device),
        mps.dtype,
        mps.config.max_bond,
        float(mps.config.cutoff),
        bool(fuse_single_qubit),
    )
    program_key = ("mps", signature)
    program = None if program_cache is None else program_cache.get(program_key)
    if program is None:
        program = CompiledMPSProgram.compile(
            instructions,
            signature=signature,
            fuse_single_qubit=fuse_single_qubit,
        )
        if program_cache is not None:
            program_cache[program_key] = program

    for operation in program.operations:
        group = operation.instruction_indices
        instruction = instructions[group[0]]
        if operation.kind == "adjacent_two_bucket":
            if not spatial_bucket:
                for index in group:
                    mps.apply_instruction(instructions[index], parameter_bindings)
                continue
            matrices = [
                gate_matrix(
                    instructions[index],
                    bsz=mps.bsz,
                    device=mps.device,
                    dtype=mps.dtype,
                    parameter_bindings=parameter_bindings,
                )
                for index in group
            ]
            mps.apply_two_bucket(
                matrices,
                [min(instructions[index].wires) for index in group],
            )
            continue
        if len(group) == 1:
            mps.apply_instruction(instruction, parameter_bindings)
            continue
        matrix = gate_matrix(
            instruction,
            bsz=mps.bsz,
            device=mps.device,
            dtype=mps.dtype,
            parameter_bindings=parameter_bindings,
        )
        for index in group[1:]:
            matrix = _compose_one_qubit_matrices(
                gate_matrix(
                    instructions[index],
                    bsz=mps.bsz,
                    device=mps.device,
                    dtype=mps.dtype,
                    parameter_bindings=parameter_bindings,
                ),
                matrix,
                bsz=mps.bsz,
            )
        mps.apply_one(matrix, int(instruction.wires[0]))
    return mps
