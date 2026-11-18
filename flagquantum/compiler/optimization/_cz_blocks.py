"""Bounded target-CZ block synthesis on native IR, using the existing KAK engine.

This private pass is target-dependent and is not part of generic ``optimize``.
It follows the collect/consolidate/resynthesize pattern used by Qiskit and Cirq;
no external IR or compiler is loaded. Numeric blocks are compared in a common
ordered basis, so reversed operands retain their control/target meaning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ...core.ir import CircuitIR, Instruction
from ..lowering.basis_translation import translate
from .two_qubit_optimization import _IDENTITY, Matrix, _lifted_matrix, _matmul
from .two_qubit_synthesis import synthesize_two_qubit

# Bound matrix-product error and synthesis work per block. Larger programs are
# streamed as multiple blocks; this is not a circuit-size admission limit.
_MAX_BLOCK_OPERATIONS = 64
_MAX_ADDITIONAL_OPERATIONS = 32
_BASIS = frozenset(("rz", "rx", "cz"))


def _matrix(instruction: Instruction, pair: tuple[int, int]) -> Matrix | None:
    if instruction.metadata or instruction.matrix is not None:
        # Annotations can carry conditions or routing-layout evidence; neither
        # may disappear into a numeric block. Matrix-valued gradients need their
        # own admission, rather than implicit scalar extraction here.
        return None
    if len(instruction.wires) == 2 and instruction.wires == pair[::-1]:
        local = _lifted_matrix(instruction, pair[::-1])
        if local is None:
            return None
        permutation = (0, 2, 1, 3)
        return [[local[i][j] for j in permutation] for i in permutation]
    return _lifted_matrix(instruction, pair)


def _product(
    instructions: Sequence[Instruction], pair: tuple[int, int]
) -> Matrix | None:
    result = _IDENTITY
    for instruction in instructions:
        matrix = _matrix(instruction, pair)
        if matrix is None:
            return None
        result = _matmul(matrix, result)
    return result


def _verified(source: Matrix, candidate: Matrix) -> bool:
    row, column = max(
        ((i, j) for i in range(4) for j in range(4)),
        key=lambda index: abs(candidate[index[0]][index[1]]),
    )
    pivot = candidate[row][column]
    if abs(pivot) < 1e-15:
        return False
    phase = source[row][column] / pivot
    return abs(abs(phase) - 1.0) <= 1e-11 and all(
        abs(source[i][j] - phase * candidate[i][j]) <= 1e-11
        for i in range(4)
        for j in range(4)
    )


def _candidate(
    block: Sequence[Instruction], pair: tuple[int, int]
) -> tuple[Instruction, ...] | None:
    baseline: list[Instruction] = []
    for instruction in block:
        if instruction.name in _BASIS:
            baseline.append(instruction)
            continue
        leaves = translate(
            instruction,
            can_run=lambda item: item.name in _BASIS,
            z_rotation="rz",
            pulse_opcode="rx",
        )
        if leaves is None or any(item.name not in _BASIS for item in leaves):
            return None
        baseline.extend(leaves)
    cz_count = sum(item.name == "cz" for item in baseline)
    if cz_count < 2:
        return None
    product = _product(block, pair)
    if product is None:
        return None
    candidate = synthesize_two_qubit(
        product, qubits=pair, entangler="cz", z_rotation="rz", pulse_opcode="rx"
    )
    if (
        candidate is None
        or any(item.name not in _BASIS for item in candidate)
        or len(candidate) > len(baseline) + _MAX_ADDITIONAL_OPERATIONS
        or sum(item.name == "cz" for item in candidate) >= cz_count
    ):
        return None
    # Bounded numerical synthesis is a proposal, not its own acceptance oracle.
    reconstructed = _product(candidate, pair)
    if reconstructed is None or not _verified(product, reconstructed):
        return None
    return candidate


def resynthesize_cz_blocks(program: CircuitIR) -> CircuitIR:
    """Reduce full-CZ cost of numeric two-qubit blocks, up to global phase.

    This private pass is admitted explicitly through the existing compiler pass
    extension point. It assumes a full-CZ plus arbitrary-one-qubit target; it must
    not run in a target-independent or phase-sensitive controlled construction.
    It preserves final placement and never absorbs a logical SWAP into a layout.
    Annotated operations, trainable/batched values, custom matrices, nonunitaries
    and wider operations form boundaries. Routing SWAP evidence is kept intact.
    """
    open_blocks: dict[tuple[int, int], list[int]] = {}
    wire_blocks: dict[int, tuple[int, int]] = {}
    changed = False

    def close(pair: tuple[int, int]) -> None:
        nonlocal changed
        positions = open_blocks.pop(pair)
        for wire in pair:
            del wire_blocks[wire]
        block = [program.instructions[index] for index in positions]
        if len(block) < 2:
            return
        candidate = _candidate(block, pair)
        if candidate is None:
            return
        replacements[positions[0]] = candidate
        for index in positions[1:]:
            replacements[index] = ()
        changed = True

    replacements: dict[int, tuple[Instruction, ...]] = {}
    for index, instruction in enumerate(program.instructions):
        wires = set(instruction.wires)
        touching = (
            {wire_blocks[wire] for wire in wires if wire in wire_blocks}
            if wires
            else set(open_blocks)
        )
        joining = next(iter(touching)) if len(touching) == 1 else None
        readable = (
            joining is not None
            and wires <= set(joining)
            and _matrix(instruction, joining) is not None
        )
        for pair in touching:
            # A zero-qubit directive is conservatively a global boundary.
            if not wires or pair != joining or not readable:
                close(pair)
        if joining is not None and readable and joining in open_blocks:
            open_blocks[joining].append(index)
            if len(open_blocks[joining]) == _MAX_BLOCK_OPERATIONS:
                close(joining)
        elif len(instruction.wires) == 2:
            left, right = instruction.wires
            pair = (min(left, right), max(left, right))
            if _matrix(instruction, pair) is not None:
                open_blocks[pair] = [index]
                for wire in pair:
                    wire_blocks[wire] = pair
    for pair in tuple(open_blocks):
        close(pair)
    if not changed:
        return program
    return replace(
        program,
        instructions=tuple(
            leaf
            for index, instruction in enumerate(program.instructions)
            for leaf in replacements.get(index, (instruction,))
        ),
    )
