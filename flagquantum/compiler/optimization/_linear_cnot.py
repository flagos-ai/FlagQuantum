# Portions adapted from Qiskit 2.6.2, crates/synthesis/src/linear/pmh.rs.
# (C) Copyright IBM 2024
# Licensed under the Apache License, Version 2.0. See the repository LICENSE or
# https://www.apache.org/licenses/LICENSE-2.0.
# Modified for FlagQuantum: integer row masks, native IR block boundaries,
# exact binary-map verification, strict gate reduction and depth admission.

"""Private CNOT block synthesis; no default pass registration or routing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ...core.ir import CircuitIR, Instruction
from ...errors import CompilationError

if TYPE_CHECKING:
    from ..topology.coupling import CouplingMap

_MAX_BLOCK_OPERATIONS = 256


def _linear_matrix(
    operations: Sequence[tuple[int, int]], width: int
) -> tuple[int, ...]:
    rows = [1 << wire for wire in range(width)]
    for control, target in operations:
        if not 0 <= control < width or not 0 <= target < width or control == target:
            raise ValueError("CNOT operands must be distinct matrix indices")
        rows[target] ^= rows[control]
    return tuple(rows)


def _lower_eliminate(
    rows: list[int], section_size: int, operation_limit: int | None = None
) -> list[tuple[int, int]] | None:
    operations: list[tuple[int, int]] = []
    width = len(rows)
    for start in range(0, width, section_size):
        stop = min(start + section_size, width)
        mask = ((1 << (stop - start)) - 1) << start
        patterns: dict[int, int] = {}
        for row in range(start, width):
            pattern = rows[row] & mask
            if not pattern:
                continue
            if pattern in patterns:
                control = patterns[pattern]
                rows[row] ^= rows[control]
                operations.append((control, row))
                if operation_limit is not None and len(operations) >= operation_limit:
                    return None
            else:
                patterns[pattern] = row
        for column in range(start, stop):
            diagonal = bool(rows[column] & (1 << column))
            for row in range(column + 1, width):
                if rows[row] & (1 << column):
                    if not diagonal:
                        rows[column] ^= rows[row]
                        operations.append((row, column))
                        if (
                            operation_limit is not None
                            and len(operations) >= operation_limit
                        ):
                            return None
                        diagonal = True
                    rows[row] ^= rows[column]
                    operations.append((column, row))
                    if (
                        operation_limit is not None
                        and len(operations) >= operation_limit
                    ):
                        return None
                # Qiskit's back reduction avoids a dense pivot growing later rows.
                if (rows[column] & rows[row]).bit_count() > 1:
                    rows[column] ^= rows[row]
                    operations.append((row, column))
                    if (
                        operation_limit is not None
                        and len(operations) >= operation_limit
                    ):
                        return None
    return operations


def _pmh_synthesize(
    matrix: tuple[int, ...], *, operation_limit: int | None = None
) -> tuple[tuple[int, int], ...] | None:
    """Adapt Qiskit's PMH row/column elimination without external matrix types."""
    width = len(matrix)
    if any(type(row) is not int or not 0 <= row < 1 << width for row in matrix):
        raise ValueError("Linear rows must be binary masks within the square matrix")
    if operation_limit is not None and (
        type(operation_limit) is not int or operation_limit < 1
    ):
        raise ValueError("Linear synthesis operation limit must be a positive integer")
    if not width:
        return ()
    section_size = max(2, math.floor(0.56 * math.log2(width)))
    rows = list(matrix)
    lower = _lower_eliminate(rows, section_size, operation_limit)
    if lower is None:
        return None
    transposed = [
        sum(((rows[row] >> column) & 1) << row for row in range(width))
        for column in range(width)
    ]
    upper = _lower_eliminate(
        transposed,
        section_size,
        None if operation_limit is None else operation_limit - len(lower),
    )
    if upper is None:
        return None
    if transposed != [1 << wire for wire in range(width)]:
        raise ValueError("Linear synthesis requires an invertible binary matrix")
    return tuple((target, control) for control, target in upper) + tuple(
        reversed(lower)
    )


def _rewrite_block(
    block: list[Instruction], coupling_map: CouplingMap | None = None
) -> tuple[Instruction, ...]:
    if len(block) < 2:
        return tuple(block)
    physical = sorted({wire for item in block for wire in item.wires})
    local = {wire: index for index, wire in enumerate(physical)}
    source = tuple((local[item.wires[0]], local[item.wires[1]]) for item in block)
    matrix = _linear_matrix(source, len(physical))
    candidate = _pmh_synthesize(matrix, operation_limit=len(block))
    if candidate is None:
        return tuple(block)
    if _linear_matrix(candidate, len(physical)) != matrix:
        raise CompilationError("CNOT synthesis changed the exact binary transformation")
    if coupling_map is not None and any(
        not coupling_map.has_edge(physical[control], physical[target])
        for control, target in candidate
    ):
        return tuple(block)
    return tuple(
        Instruction("cx", (physical[control], physical[target]))
        for control, target in candidate
    )


def _two_body_depth(instructions: Sequence[Instruction], width: int) -> int:
    levels = [0] * width
    for item in instructions:
        if len(item.wires) == 2:
            level = max(levels[wire] for wire in item.wires) + 1
            for wire in item.wires:
                levels[wire] = level
    return max(levels, default=0)


def resynthesize_cnot_blocks(
    program: CircuitIR, *, coupling_map: CouplingMap | None = None
) -> CircuitIR:
    """Propose exact linear rewrites through the existing pass hook.

    Only unannotated, parameter-free CX runs join a block. Every emitted block
    must strictly reduce CX count and reproduce the exact binary map; the whole
    output must not increase two-body dependency depth. No layout is absorbed.
    A supplied undirected coupling map admits only proposals on its edges;
    it never routes a proposal. With no map, connectivity is all-to-all.
    This private pass is not enabled by default. It does not certify the cost
    of later routing when other source operations still require movement.
    """
    if coupling_map is not None and coupling_map.n_qubits < program.n_wires:
        raise ValueError("Linear synthesis coupling map is smaller than the program")
    output: list[Instruction] = []
    block: list[Instruction] = []
    for item in program.instructions:
        if (
            item.name == "cx"
            and len(item.wires) == 2
            and not item.params
            and not item.metadata
            and item.matrix is None
        ):
            block.append(item)
            if len(block) == _MAX_BLOCK_OPERATIONS:
                output.extend(_rewrite_block(block, coupling_map))
                block.clear()
        else:
            output.extend(_rewrite_block(block, coupling_map))
            block.clear()
            output.append(item)
    output.extend(_rewrite_block(block, coupling_map))
    if len(output) == len(program.instructions) or _two_body_depth(
        output, program.n_wires
    ) > _two_body_depth(program.instructions, program.n_wires):
        return program
    return replace(program, instructions=tuple(output))
