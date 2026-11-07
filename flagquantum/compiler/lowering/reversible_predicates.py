"""Bind Core Boolean predicates to native positive masks and owned scratch."""

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
    """Positive native slots and the prefix defining owned predicate scratch."""

    prefix: tuple[Operation, ...]
    slots: tuple[int, ...]
    scratch_width: int


def _complement(bit: int, scratch: int) -> tuple[Operation, ...]:
    return (
        Operation("bit_store0", (), result_bit=scratch),
        Operation("bit_invert", (), result_bit=scratch),
        Operation("bit_invert", (), result_bit=scratch, condition_bit=bit),
    )


def _disjunction(
    clauses: tuple[tuple[tuple[int, int], ...], ...], first_scratch: int
) -> PredicateBinding:
    negative = tuple(
        dict.fromkeys(bit for clause in clauses for bit, value in clause if not value)
    )
    complements = {bit: first_scratch + index for index, bit in enumerate(negative)}
    accumulator = first_scratch + len(negative)
    unset = accumulator + 1
    operations = [
        op for bit, scratch in complements.items() for op in _complement(bit, scratch)
    ]
    operations.append(Operation("bit_store0", (), result_bit=accumulator))
    for clause in clauses:
        operations.extend(_complement(accumulator, unset))
        guards = (unset, *(bit if value else complements[bit] for bit, value in clause))
        operations.extend(
            Operation("push_condition", (), condition_bit=guard) for guard in guards
        )
        operations.append(Operation("bit_invert", (), result_bit=accumulator))
        operations.extend(Operation("pop_condition", ()) for _ in guards)
    return PredicateBinding(tuple(operations), (accumulator,), len(negative) + 2)


def bind_predicate(instruction: Instruction, *, first_scratch: int) -> PredicateBinding:
    """Use scratch above all original program slots, reusable after scope exit."""
    clauses = instruction_condition_clauses(instruction)
    if len(clauses) > 1:
        return _disjunction(clauses, first_scratch)
    operations: list[Operation] = []
    slots: list[int] = []
    width = 0
    for bit, value in condition_pairs(instruction):
        if value:
            slots.append(bit)
            continue
        scratch = first_scratch + width
        width += 1
        operations.extend(_complement(bit, scratch))
        slots.append(scratch)
    return PredicateBinding(tuple(operations), tuple(slots), width)
