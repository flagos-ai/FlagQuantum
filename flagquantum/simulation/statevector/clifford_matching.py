"""Compilation of exact disjoint Clifford matching layers."""

from __future__ import annotations

from collections.abc import Sequence

from ...core.operator_schema import canonical_opcode
from .program import _StatevectorGateStep, _StatevectorPreCXStep


def _reorder_disjoint_clifford_matchings(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Place disjoint CZ edges before CX edges so each kind can be batched."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not _is_exact_cx_or_cz(step):
            optimized.append(step)
            index += 1
            continue

        matching: list[_StatevectorGateStep] = []
        occupied_wires: set[int] = set()
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not _is_exact_cx_or_cz(candidate):
                break
            assert isinstance(candidate, _StatevectorGateStep)
            wires = set(map(int, candidate.instruction.wires))
            if not occupied_wires.isdisjoint(wires):
                break
            matching.append(candidate)
            occupied_wires.update(wires)
            cursor += 1

        cz_steps = [
            item for item in matching if canonical_opcode(item.instruction.name) == "cz"
        ]
        cx_steps = [
            item for item in matching if canonical_opcode(item.instruction.name) == "cx"
        ]
        optimized.extend((*cz_steps, *cx_steps) if cz_steps and cx_steps else matching)
        index = cursor
    return optimized


def _is_exact_cx_or_cz(step: _StatevectorPreCXStep) -> bool:
    if not isinstance(step, _StatevectorGateStep):
        return False
    instruction = step.instruction
    return (
        instruction.matrix is None
        and not instruction.params
        and canonical_opcode(instruction.name) in {"cx", "cz"}
    )
