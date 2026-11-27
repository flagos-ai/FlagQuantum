"""Numerical application of one canonical instruction to local trajectories."""

from __future__ import annotations

import torch

from flagquantum.core.ir import Instruction
from flagquantum.core.operators.schema import canonical_opcode
from flagquantum.simulation.gate_matrix import gate_matrix as _gate_matrix

from .operations import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_single_qubit_fixed,
    _statevector_layout,
)


def _apply_instruction(
    state: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor:
    """Apply one static instruction without constructing a temporary circuit."""

    name = canonical_opcode(instruction.name)
    carries_matrix = instruction.matrix is not None
    if not carries_matrix and name in {"x", "cx", "swap"}:
        return _apply_fixed_permutation(state, name, instruction.wires, n_wires)
    if not carries_matrix and name == "y":
        return _apply_single_qubit_fixed(
            state,
            name,
            instruction.wires[0],
            n_wires,
        )
    matrix = _gate_matrix(
        instruction,
        bsz=state.shape[0],
        device=state.device,
        dtype=state.dtype,
        parameter_bindings=None,
    )
    apply_gate = (
        _apply_diagonal_matrix
        if not carries_matrix and name in _DIAGONAL_STATEVECTOR_GATES
        else _apply_matrix
    )
    return apply_gate(
        state,
        matrix,
        instruction.wires,
        n_wires,
        layout=_statevector_layout(n_wires, instruction.wires),
    )
