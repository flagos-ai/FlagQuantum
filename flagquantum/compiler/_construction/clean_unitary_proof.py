"""Matrix proof for small verified register blocks with clean work bits."""

from collections.abc import Sequence

import torch

from ...core.ir import Instruction
from .._hybrid.model import HybridProgram
from ..one_qubit_optimization import _instruction_matrix as one_qubit_matrix
from ..two_qubit_optimization import _instruction_matrix as two_qubit_matrix
from .profile import _fail
from .register_lowering import schedule_register_program


def _matrix(gate: Instruction) -> torch.Tensor | None:
    if gate.matrix is not None or gate.metadata:
        return None
    if len(gate.wires) == 1:
        matrix = one_qubit_matrix(gate)
    elif len(gate.wires) == 2:
        matrix = two_qubit_matrix(gate)
    elif gate.name == "ccx" and len(gate.wires) == 3:
        identity = torch.eye(8, dtype=torch.complex128)
        return identity[[0, 1, 2, 3, 4, 5, 7, 6]]
    else:
        return None
    return None if matrix is None else torch.tensor(matrix, dtype=torch.complex128)


def verify_clean_unitary(program: HybridProgram, width: int) -> torch.Tensor:
    """Prove the full logical unitary on a zero-workspace entry subspace."""
    if not 1 <= width <= 5:
        _fail("lowering.proof", "clean unitary proof admits up to five logical bits")
    n_wires = width + 2
    candidate = schedule_register_program(
        program, n_qubits=n_wires, max_operations=4096
    )
    if sum(map(len, candidate.registers.values())) != width:
        _fail("lowering.proof", "proof output width disagrees with its declared ports")

    dimension = 1 << width
    columns = torch.eye(1 << n_wires, dtype=torch.complex128)[:, ::4].contiguous()
    columns = _replay_columns(columns, candidate.circuit.instructions, n_wires)
    blocks = columns.reshape(dimension, 4, dimension)
    if blocks[:, 1:, :].abs().max().item() > 1e-12:
        _fail(
            "lowering.proof", "reversible block does not return its work bits to zero"
        )
    logical = blocks[:, 0, :]
    identity = torch.eye(dimension, dtype=torch.complex128)
    if not torch.allclose(logical.mH @ logical, identity, atol=1e-12, rtol=0):
        _fail("lowering.proof", "reversible block does not implement a unitary")
    return logical


def _replay_columns(
    columns: torch.Tensor, instructions: Sequence[Instruction], n_qubits: int
) -> torch.Tensor:
    dimension = columns.shape[1]
    axes = tuple(range(n_qubits))
    for gate in instructions:
        local = _matrix(gate)
        if local is None:
            _fail("lowering.proof", f"gate {gate.name} has no bounded proof matrix")
        ordering = (
            *gate.wires,
            *(wire for wire in axes if wire not in gate.wires),
            n_qubits,
        )
        inverse = tuple(ordering.index(wire) for wire in range(n_qubits + 1))
        state = columns.reshape(*([2] * n_qubits), dimension)
        state = state.permute(ordering).reshape(local.shape[0], -1)
        columns = (local @ state).reshape(*([2] * n_qubits), dimension).permute(inverse)
    return columns


def verify_zero_preparation(program: HybridProgram, width: int) -> torch.Tensor:
    """Check one normalized logical column and its clean workspace."""
    if not 1 <= width <= 12:
        _fail(
            "lowering.proof",
            "zero preparation proof admits one through twelve logical bits",
        )
    n_qubits = width + 2
    candidate = schedule_register_program(
        program, n_qubits=n_qubits, max_operations=4096
    )
    if tuple(candidate.registers.values()) != (tuple(range(width)),):
        _fail(
            "lowering.proof",
            "preparation proof requires a single ordered logical register",
        )
    column = torch.zeros((1 << n_qubits, 1), dtype=torch.complex128)
    column[0, 0] = 1
    state = _replay_columns(column, candidate.circuit.instructions, n_qubits).reshape(
        -1, 4
    )
    logical = state[:, 0]
    if (
        not torch.isfinite(state).all().item()
        or state[:, 1:].abs().max().item() > 1e-12
        or abs(float(logical.abs().square().sum()) - 1.0) > 1e-12
    ):
        _fail(
            "lowering.proof",
            "preparation proof requires a normalized state and clean workspace",
        )
    return logical
