"""Lower instructions with explicit physical component and feedback ranges."""

from dataclasses import dataclass, field, replace

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import CircuitIR, Instruction
from .._component_trace.trace import ComponentCall
from ..artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from .reversible_phase import lower_reversible_phase


@dataclass
class PhysicalTrace:
    """Own source ranges in instruction offsets and implicit zero slots."""

    calls: list[ComponentCall] = field(default_factory=list)
    feedback: list[tuple[int, int, int]] = field(default_factory=list)
    implicit_zero_bits: set[int] = field(default_factory=set)
    recycled_bits: frozenset[int] = frozenset()


def _feedback_order(
    operations: tuple[Operation, ...], result_start: int
) -> tuple[Operation, ...]:
    """Place the HMR result outside the coefficient test for a phase gate."""
    result: list[Operation] = []
    i = 0
    while i < len(operations):
        span = operations[i : i + 3]
        if len(span) == 3:
            enter, gate, leave = span
            if (
                enter.opcode == "push_condition"
                and enter.condition_bit is not None
                and enter.condition_bit < result_start
                and gate.condition_bit is not None
                and gate.condition_bit >= result_start
                and leave.opcode == "pop_condition"
            ):
                result.extend(
                    (
                        replace(enter, condition_bit=gate.condition_bit),
                        replace(gate, condition_bit=enter.condition_bit),
                        leave,
                    )
                )
                i += 3
                continue
        result.append(operations[i])
        i += 1
    return tuple(result)


def _scoped_records(
    operations: tuple[Operation, ...], bit: int
) -> tuple[Operation, ...]:
    """Hold one measurement guard across the whole source feedback body."""
    result = [Operation("push_condition", (), condition_bit=bit)]
    i = 0
    while i < len(operations):
        gate = operations[i]
        if gate.opcode == "push_condition":
            inner, leave = operations[i + 1 : i + 3]
            assert leave.opcode == "pop_condition"
            if gate.condition_bit == bit:
                result.append(inner)
            else:
                assert inner.condition_bit == bit
                result.append(replace(inner, condition_bit=gate.condition_bit))
            i += 3
        else:
            assert gate.condition_bit == bit
            result.append(replace(gate, condition_bit=None))
            i += 1
    return (*result, Operation("pop_condition", ()))


def lower_physical_trace(
    gates: tuple[Instruction, ...],
    inputs: int,
    fallback_wire: int,
    export: PhysicalTrace,
    *,
    reorder_feedback: bool = True,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    # Definitions used by the IR validator. The physical input and fresh vent
    # slots already belong to the format's initial classical storage.
    known = (*range(inputs), *sorted(export.implicit_zero_bits))
    setup = tuple(ClassicalXor((), bit).instruction() for bit in known)
    offsets: dict[int, int] = {}
    width = 1 + max((q for gate in gates for q in gate.wires), default=fallback_wire)
    lowered = lower_reversible_phase(
        CircuitIR(width, (*setup, *gates)), operation_offsets=offsets
    )
    ordered = (
        _feedback_order(lowered.operations, inputs)
        if reorder_feedback
        else lowered.operations
    )
    operations: list[Operation] = []
    boundary: dict[int, int] = {}
    cursor = len(setup)
    for start, stop, bit in sorted(export.feedback):
        a, b = offsets[start + len(setup)], offsets[stop + len(setup)]
        for pos in range(cursor, a + 1):
            boundary[pos] = len(operations) + pos - cursor
        operations.extend(ordered[cursor:a])
        operations.extend(_scoped_records(ordered[a:b], bit))
        cursor = b
    for pos in range(cursor, len(ordered) + 1):
        boundary[pos] = len(operations) + pos - cursor
    operations.extend(ordered[cursor:])

    def rebind(call: ComponentCall) -> ComponentCall:
        return replace(
            call,
            start=boundary[offsets[call.start + len(setup)]],
            stop=boundary[offsets[call.stop + len(setup)]],
            children=tuple(rebind(child) for child in call.children),
        )

    return (
        ReversiblePhaseProgram(
            width,
            max(inputs, lowered.n_bits),
            tuple(operations),
            tuple(range(inputs)) + tuple(sorted(export.implicit_zero_bits)),
        ),
        tuple(rebind(call) for call in export.calls),
    )
