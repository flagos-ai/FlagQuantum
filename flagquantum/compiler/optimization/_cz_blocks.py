"""Bounded target-CZ block synthesis on native IR, using the existing KAK engine.

This private pass is target-dependent and is not part of generic ``optimize``.
It follows the collect/consolidate/resynthesize pattern used by Qiskit and Cirq;
no external IR or compiler is loaded. Numeric blocks are compared in a common
ordered basis, so reversed operands retain their control/target meaning.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from ..lowering.basis_translation import translate
from ._cz_local_cleanup import compact_native_cz_locals
from .commutation_cancellation import merge_commuting_rotations
from .two_qubit_optimization import _IDENTITY, Matrix, _lifted_matrix, _matmul
from .two_qubit_synthesis import synthesize_two_qubit

# Bound matrix-product error and synthesis work per block. Larger programs are
# streamed as multiple blocks; this is not a circuit-size admission limit.
_MAX_BLOCK_OPERATIONS = 64
_MAX_ADDITIONAL_OPERATIONS = 32
_BASIS = frozenset(("rz", "rx", "cz"))
RejectionReason = Literal[
    "not_lower_cz_cost",
    "synthesis_unavailable",
    "residual_rejected",
    "instruction_growth_rejected",
]
MatrixReader = Callable[[Instruction, tuple[int, int]], Matrix | None]


def _matrix(instruction: Instruction, pair: tuple[int, int]) -> Matrix | None:
    if instruction.metadata or instruction.matrix is not None:
        # Annotations can carry conditions or routing-layout evidence; neither
        # may disappear into a numeric block. Matrix-valued gradients need their
        # own admission, rather than implicit scalar extraction here.
        return None
    return _numeric_matrix(instruction, pair)


def _numeric_matrix(instruction: Instruction, pair: tuple[int, int]) -> Matrix | None:
    if instruction.matrix is not None:
        return None
    if len(instruction.wires) == 2 and instruction.wires == pair[::-1]:
        local = _lifted_matrix(instruction, pair[::-1])
        if local is None:
            return None
        permutation = (0, 2, 1, 3)
        return [[local[i][j] for j in permutation] for i in permutation]
    return _lifted_matrix(instruction, pair)


def _product(
    instructions: Sequence[Instruction],
    pair: tuple[int, int],
    *,
    matrix_reader: MatrixReader = _matrix,
) -> Matrix | None:
    result = _IDENTITY
    for instruction in instructions:
        matrix = matrix_reader(instruction, pair)
        if matrix is None:
            return None
        result = _matmul(matrix, result)
    return result


def _equivalence_witness(
    source: Matrix, candidate: Matrix
) -> tuple[complex, float] | None:
    row, column = max(
        ((i, j) for i in range(4) for j in range(4)),
        key=lambda index: abs(candidate[index[0]][index[1]]),
    )
    pivot = candidate[row][column]
    if abs(pivot) < 1e-15:
        return None
    phase = source[row][column] / pivot
    residual = max(
        abs(source[i][j] - phase * candidate[i][j]) for i in range(4) for j in range(4)
    )
    if abs(abs(phase) - 1.0) <= 1e-11 and residual <= 1e-11:
        return phase, residual
    return None


def _verified(source: Matrix, candidate: Matrix) -> bool:
    return _equivalence_witness(source, candidate) is not None


def _candidate(
    block: Sequence[Instruction],
    pair: tuple[int, int],
    *,
    max_output_operations: int | None = None,
    on_rejection: Callable[[RejectionReason], None] | None = None,
    matrix_reader: MatrixReader = _matrix,
) -> tuple[Instruction, ...] | None:
    def reject(reason: RejectionReason) -> None:
        if on_rejection is not None:
            on_rejection(reason)

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
            reject("synthesis_unavailable")
            return None
        baseline.extend(leaves)
    cz_count = sum(item.name == "cz" for item in baseline)
    if cz_count < 2:
        reject("not_lower_cz_cost")
        return None
    product = _product(block, pair, matrix_reader=matrix_reader)
    if product is None:
        reject("synthesis_unavailable")
        return None
    candidate = synthesize_two_qubit(
        product, qubits=pair, entangler="cz", z_rotation="rz", pulse_opcode="rx"
    )
    if candidate is None or any(item.name not in _BASIS for item in candidate):
        reject("synthesis_unavailable")
        return None
    limit = len(baseline) + _MAX_ADDITIONAL_OPERATIONS
    if max_output_operations is not None:
        limit = min(limit, max_output_operations)
    if len(candidate) > limit:
        reject("instruction_growth_rejected")
        return None
    if sum(item.name == "cz" for item in candidate) >= cz_count:
        reject("not_lower_cz_cost")
        return None
    # This pass explicitly assumes arbitrary one-qubit target rotations. Keep
    # the generic synthesis entry's fixed-pulse contract unchanged, and compact
    # only bounded proposals that already satisfy the raw resource limits.
    candidate = compact_native_cz_locals(candidate, pair)
    # Euler cleanup exposes rotations separated by commuting CZ gates. Reuse
    # the existing proof-based merge on this bounded, numeric proposal only.
    candidate = merge_commuting_rotations(
        CircuitIR(max(pair) + 1, candidate)
    ).instructions
    # Bounded numerical synthesis is a proposal, not its own acceptance oracle.
    reconstructed = _product(candidate, pair)
    if reconstructed is None or not _verified(product, reconstructed):
        reject("residual_rejected")
        return None
    return candidate


def _collect_blocks(
    program: CircuitIR,
    readable: Callable[[int, tuple[int, int]], bool],
    *,
    barrier: Callable[[Instruction], bool] | None = None,
    identity_reader: MatrixReader | None = None,
) -> Iterator[tuple[tuple[int, int], tuple[int, ...]]]:
    """Collect bounded pair blocks, checking identity prefixes only at capacity.

    Leading locals still join a closed prefix. The caller must synthesize and
    verify that complete block; prefix detection only proposes a split point.
    """
    open_blocks: dict[tuple[int, int], list[int]] = {}
    wire_blocks: dict[int, tuple[int, int]] = {}
    claimed: set[int] = set()

    def close(pair: tuple[int, int]) -> tuple[int, ...]:
        positions = open_blocks.pop(pair)
        for wire in pair:
            del wire_blocks[wire]
        # Use the existing two-qubit collector's exclusive leading-member rule.
        # Draw locals in only after closing, so they consume remaining capacity
        # without splitting the established entangling block earlier.
        leading: list[int] = []
        back = positions[0] - 1
        while (
            back >= 0
            and back not in claimed
            and len(leading) + len(positions) < _MAX_BLOCK_OPERATIONS
        ):
            instruction = program.instructions[back]
            if len(instruction.wires) != 1 or not readable(back, pair):
                break
            leading.append(back)
            back -= 1
        positions = [*reversed(leading), *positions]
        claimed.update(leading)
        return tuple(positions)

    for index, instruction in enumerate(program.instructions):
        if barrier is not None and barrier(instruction):
            for pair in tuple(open_blocks):
                yield pair, close(pair)
            continue
        wires = set(instruction.wires)
        touching = (
            {wire_blocks[wire] for wire in wires if wire in wire_blocks}
            if wires
            else set(open_blocks)
        )
        joining = next(iter(touching)) if len(touching) == 1 else None
        joins = (
            joining is not None and wires <= set(joining) and readable(index, joining)
        )
        for pair in touching:
            # A zero-qubit directive is conservatively a global boundary.
            if not wires or pair != joining or not joins:
                yield pair, close(pair)
        if joining is not None and joins and joining in open_blocks:
            pair = joining
        elif len(instruction.wires) == 2:
            left, right = instruction.wires
            pair = (min(left, right), max(left, right))
            if readable(index, pair):
                open_blocks[pair] = []
                for wire in pair:
                    wire_blocks[wire] = pair
            else:
                continue
        else:
            continue
        open_blocks[pair].append(index)
        claimed.add(index)
        if len(open_blocks[pair]) == _MAX_BLOCK_OPERATIONS:
            positions = open_blocks[pair]
            split = len(positions)
            if identity_reader is not None:
                product = _IDENTITY
                entanglers = 0
                # Short blocks keep their existing collection fast path. Only
                # a forced split needs a search for a better bounded boundary.
                for offset, position in enumerate(positions, 1):
                    member = program.instructions[position]
                    matrix = identity_reader(member, pair)
                    if matrix is None:
                        break
                    product = _matmul(matrix, product)
                    entanglers += len(member.wires) == 2
                    if entanglers >= 2 and _verified(product, _IDENTITY):
                        split = offset
                        break
            suffix = positions[split:]
            open_blocks[pair] = positions[:split]
            yield pair, close(pair)
            if suffix:
                open_blocks[pair] = suffix
                for wire in pair:
                    wire_blocks[wire] = pair
    for pair in tuple(open_blocks):
        yield pair, close(pair)


def resynthesize_cz_blocks(program: CircuitIR) -> CircuitIR:
    """Reduce full-CZ cost of numeric two-qubit blocks, up to global phase.

    This private pass is admitted explicitly through the existing compiler pass
    extension point. It assumes a full-CZ plus arbitrary-one-qubit target; it must
    not run in a target-independent or phase-sensitive controlled construction.
    It preserves final placement and never absorbs a logical SWAP into a layout.
    Annotated operations, trainable/batched values, custom matrices, nonunitaries
    and wider operations form boundaries. Routing SWAP evidence is kept intact.
    """
    replacements: dict[int, tuple[Instruction, ...]] = {}
    for pair, positions in _collect_blocks(
        program,
        lambda index, pair: _matrix(program.instructions[index], pair) is not None,
    ):
        if len(positions) < 2:
            continue
        candidate = _candidate(
            [program.instructions[index] for index in positions], pair
        )
        if candidate is None:
            continue
        replacements[positions[0]] = candidate
        for index in positions[1:]:
            replacements[index] = ()
    if not replacements:
        return program
    return replace(
        program,
        instructions=tuple(
            leaf
            for index, instruction in enumerate(program.instructions)
            for leaf in replacements.get(index, (instruction,))
        ),
    )
