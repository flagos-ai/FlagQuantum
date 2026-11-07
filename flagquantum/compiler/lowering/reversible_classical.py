"""Legalize classical parity while retaining atomic conditional reads."""

from ...core.classical_dataflow import parse_classical_xor
from ...core.ir import Instruction
from ..artifacts.reversible_phase import ReversiblePhaseOperation as Operation


def lower_classical_assignment(
    instruction: Instruction, *, scopes: tuple[int, ...]
) -> tuple[Operation, ...]:
    """Snapshot guards before expanding an old-read XOR assignment."""
    assignment = parse_classical_xor(instruction)
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
