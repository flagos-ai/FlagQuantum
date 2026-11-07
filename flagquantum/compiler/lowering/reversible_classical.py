"""Legalize classical parity while retaining atomic conditional reads."""

from ...core.classical_control import instruction_condition_clauses
from ...core.classical_dataflow import parse_classical_xor
from ...core.ir import Instruction
from ...errors import CapabilityError
from ..artifacts.reversible_phase import ReversiblePhaseOperation as Operation


def positive_condition_slots(instruction: Instruction) -> tuple[int, ...]:
    """Admit one canonical positive conjunction for the native target."""
    clauses = instruction_condition_clauses(instruction)
    if not clauses:
        return ()
    if len(clauses) != 1 or any(value != 1 for _, value in clauses[0]):
        raise CapabilityError(
            "reversible-phase lowering requires one positive condition clause"
        )
    return tuple(slot for slot, _ in clauses[0])


def lower_classical_assignment(instruction: Instruction) -> tuple[Operation, ...]:
    """Snapshot guards before expanding an old-read XOR assignment."""
    assignment = parse_classical_xor(instruction)
    scopes = positive_condition_slots(instruction)
    operations = [
        Operation("push_condition", (), condition_bit=slot) for slot in scopes
    ]
    if assignment.sources.count(assignment.target) % 2 == 0:
        operations.append(
            Operation(
                "bit_invert" if scopes else "bit_store0",
                (),
                result_bit=assignment.target,
                condition_bit=assignment.target if scopes else None,
            )
        )
    operations.extend(
        Operation("bit_invert", (), result_bit=assignment.target, condition_bit=source)
        for source in assignment.sources
        if source != assignment.target
    )
    if assignment.constant:
        operations.append(Operation("bit_invert", (), result_bit=assignment.target))
    operations.extend(Operation("pop_condition", ()) for _ in scopes)
    return tuple(operations)
