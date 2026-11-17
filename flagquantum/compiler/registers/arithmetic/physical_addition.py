# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Export mixed addition with its physical call and feedback boundaries."""

from dataclasses import replace

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import CircuitIR, Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.reversible_phase import lower_reversible_phase
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .measured_carry_addition import measured_carry_addition_ir
from .mixed_addition import AdditionExport, mixed_addition_ir


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
        if gate.opcode == "push_condition" and gate.condition_bit == bit:
            inner, leave = operations[i + 1 : i + 3]
            assert leave.opcode == "pop_condition"
            result.append(inner)
            i += 3
        else:
            assert gate.condition_bit == bit
            result.append(replace(gate, condition_bit=None))
            i += 1
    return (*result, Operation("pop_condition", ()))


def _physical_program(
    gates: tuple[Instruction, ...],
    inputs: int,
    fallback_wire: int,
    export: AdditionExport,
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
    ordered = _feedback_order(lowered.operations, inputs)
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


def physical_classical_addition(
    target: tuple[int, ...],
    offset: tuple[ClassicalFieldCoefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    control: int | bool,
    carry_in: int | None,
    minimize_qubits: bool,
    btol: float,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind source physical classical register slots and arithmetic mode."""
    width = len(target)
    export = AdditionExport()
    if carry_in is None:
        gates = mixed_addition_ir(
            target,
            offset,
            clean=clean,
            dirty=dirty,
            control=control,
            classical_bits=tuple(range(width, 2 * width + 1)),
            minimize_qubits=minimize_qubits,
            btol=btol,
            export=export,
        )
    else:
        gates = measured_carry_addition_ir(
            target,
            offset,
            clean=clean,
            control=control,
            carry_in=carry_in,
            measurement_bit=width,
            emit_resets=True,
        )
        name = "iadd_classical_simple"
        export.calls.append(
            ComponentCall(f"c_{name}" if type(control) is int else name, 0, len(gates))
        )
    return _physical_program(gates, width, 0, export)
