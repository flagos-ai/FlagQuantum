"""Bind canonical conjunctions to native positive masks and owned complements."""

from dataclasses import dataclass

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import Instruction
from ...errors import CapabilityError
from ..artifacts.reversible_phase import ReversiblePhaseOperation as Operation


def condition_pairs(instruction: Instruction) -> tuple[tuple[int, int], ...]:
    """Admit one canonical conjunction without changing predicate polarity."""
    clauses = instruction_condition_clauses(instruction)
    if len(clauses) > 1:
        raise CapabilityError("reversible-phase lowering requires one condition clause")
    return clauses[0] if clauses else ()


@dataclass(frozen=True)
class PredicateBinding:
    """Positive native slots and the prefix defining owned negative complements."""

    prefix: tuple[Operation, ...]
    slots: tuple[int, ...]
    scratch_width: int


def bind_predicate(instruction: Instruction, *, first_scratch: int) -> PredicateBinding:
    """Use scratch above all original program slots, reusable after scope exit."""
    operations: list[Operation] = []
    slots: list[int] = []
    width = 0
    for bit, value in condition_pairs(instruction):
        if value:
            slots.append(bit)
            continue
        scratch = first_scratch + width
        width += 1
        operations.extend(
            (
                Operation("bit_store0", (), result_bit=scratch),
                Operation("bit_invert", (), result_bit=scratch),
                Operation("bit_invert", (), result_bit=scratch, condition_bit=bit),
            )
        )
        slots.append(scratch)
    return PredicateBinding(tuple(operations), tuple(slots), width)
