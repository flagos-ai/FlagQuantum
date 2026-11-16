"""Exact dense inspection of small measurement-feedback instruments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ..core.classical_dataflow import parse_classical_xor
from ..core.ir import CircuitIR, Instruction
from .gate_matrix import gate_matrix
from .statevector.operations import _apply_matrix
from .unitary import MAX_UNITARY_QUBITS, _restore_workspace, _workspace_basis

_MAX_EFFECTS = 8
_MAX_BRANCH_AMPLITUDES = 1 << 20
_ATOL = 2e-12


@dataclass(frozen=True)
class _Branch:
    amplitudes: torch.Tensor
    classical: dict[int, int]


def _split_effect(
    branch: _Branch, instruction: Instruction, indices: torch.Tensor, width: int
) -> list[_Branch]:
    mask = 1 << (width - instruction.wires[0] - 1)
    outcomes = (indices & mask) != 0
    result: list[_Branch] = []
    for bit in (0, 1):
        amplitudes = branch.amplitudes * (outcomes == bit)
        if not torch.any(amplitudes):
            continue
        classical = branch.classical
        if instruction.name == "measure":
            classical = {**classical, instruction.metadata["classical_bit"]: bit}
        elif bit:
            amplitudes = amplitudes.index_select(1, indices ^ mask)
        result.append(_Branch(amplitudes, classical))
    return result


def _unitary_from_branches(
    matrices: list[torch.Tensor], dimension: int
) -> torch.Tensor:
    representative: torch.Tensor | None = None
    probability = 0.0
    for matrix in matrices:
        if not torch.any(matrix):
            continue
        if representative is None:
            sample_weight = torch.linalg.vector_norm(matrix) / math.sqrt(dimension)
            representative = matrix / sample_weight
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device)
            if not torch.allclose(
                representative.mH @ representative, identity, atol=_ATOL, rtol=0
            ):
                raise ValueError("Feedback branches do not define a unitary")
        coefficient = torch.sum(representative.conj() * matrix) / dimension
        if not torch.allclose(matrix, coefficient * representative, atol=_ATOL, rtol=0):
            raise ValueError("Feedback branches do not define the same unitary")
        probability += float(coefficient.abs().square())
    if representative is None or not math.isclose(
        probability, 1.0, abs_tol=_ATOL, rel_tol=0
    ):
        raise ValueError("Feedback branches are not probability complete")
    return representative


def _feedback_unitary(
    program: CircuitIR, logical_width: int, borrowed_wires: frozenset[int]
) -> torch.Tensor:
    """Return a logical unitary only when all exact branch matrices certify it.

    Dense cost is bounded by physical width and by the worst-case product of
    input columns, physical amplitudes and measurement/reset histories. The
    normalized first nonzero branch fixes the representative's phase.
    """
    width = program.n_wires
    effects = sum(op.name in {"measure", "reset"} for op in program.instructions)
    if (
        width > MAX_UNITARY_QUBITS
        or effects > _MAX_EFFECTS
        or not 0 < logical_width <= width
        or len(borrowed_wires) > width - logical_width
        or (1 << (width + logical_width + len(borrowed_wires) + effects))
        > _MAX_BRANCH_AMPLITUDES
    ):
        raise ValueError("Feedback body exceeds the dense branch inspection budget")
    if program.observables or program.measurements:
        raise ValueError(
            "Feedback unitary inspection requires a circuit without result requests"
        )
    validate_classical_control(program.instructions, n_qubits=width)
    admitted, rejected = _workspace_basis(width, logical_width, borrowed_wires)
    indices = torch.arange(1 << width, device="cpu")
    columns = torch.zeros(
        (admitted.numel(), indices.numel()), dtype=torch.complex128, device="cpu"
    )
    columns[torch.arange(admitted.numel(), device="cpu"), admitted] = 1
    branches = [_Branch(columns, {})]

    for instruction in program.instructions:
        if instruction.metadata.get("is_channel"):
            raise ValueError("Feedback unitary inspection requires ideal instructions")
        clauses = instruction_condition_clauses(instruction)
        effect = instruction.name in {"measure", "reset"}
        assignment = (
            parse_classical_xor(instruction)
            if instruction.name == "classical_xor"
            else None
        )
        if effect:
            if len(instruction.wires) != 1 or instruction.matrix is not None:
                raise ValueError(
                    "Feedback unitary requires a single-wire measurement or reset"
                )
            matrix = None
        elif assignment is None:
            matrix = gate_matrix(
                instruction, bsz=columns.shape[0], device="cpu", dtype=columns.dtype
            )
        else:
            matrix = None

        updated: list[_Branch] = []
        for branch in branches:
            active = not clauses or any(
                all(branch.classical[slot] == bit for slot, bit in clause)
                for clause in clauses
            )
            if not active:
                updated.append(branch)
            elif effect:
                updated.extend(_split_effect(branch, instruction, indices, width))
            elif assignment is not None:
                value = assignment.evaluate(branch.classical)
                updated.append(
                    _Branch(
                        branch.amplitudes,
                        {**branch.classical, assignment.target: value},
                    )
                )
            else:
                assert matrix is not None
                amplitudes = _apply_matrix(
                    branch.amplitudes, matrix, instruction.wires, width
                )
                updated.append(_Branch(amplitudes, branch.classical))
        branches = updated

    matrices = [
        _restore_workspace(branch.amplitudes.T, admitted, rejected, len(borrowed_wires))
        for branch in branches
    ]
    return _unitary_from_branches(matrices, 1 << logical_width)
