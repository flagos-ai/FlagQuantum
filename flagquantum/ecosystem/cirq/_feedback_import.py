"""Explicit lossless import of measured Cirq circuits and Boolean feedback.

External measurement keys end here. Native slots represent each recorded bit
and any composed Boolean predicate. The existing static conversion is separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from typing import Any

from ...compiler.registers.primitives.global_phase import (
    global_phase_exponent,
    global_phase_ir,
)
from ...core.classical_control import validate_classical_control
from ...core.classical_dataflow import ClassicalXor
from ...core.ir import CircuitIR, Instruction
from .conversion import _cirq, _lower_gate, _real_scalar, _report
from .models import CirqConversionError, CirqConversionIssue, CirqImportResult

Bit = int | bool


def _lower_feedback_gate(
    cirq: Any, gate: Any, issues: list[CirqConversionIssue], index: int
) -> tuple[str, dict[str, Any]] | None:
    """Retain exact phase powers and canonical singly controlled primitives."""
    if isinstance(gate, (cirq.ZPowGate, cirq.CZPowGate)) and gate.global_shift == 0:
        # A Z power is a phase gate, not Rz: its scalar phase is observable
        # after coherent control. Preserve exact fixed opcodes when available.
        lowered = _lower_gate(cirq, gate, [], index)
        if lowered is not None:
            return lowered
        exponent = _real_scalar(gate.exponent, issues, index, "phase power")
        if exponent is not None:
            return ("cphase" if isinstance(gate, cirq.CZPowGate) else "phase"), {
                "theta": pi * exponent
            }
        return None
    if (
        isinstance(gate, cirq.ControlledGate)
        and gate.num_controls() == 1
        and tuple(tuple(v) for v in gate.control_values) == ((1,),)
    ):
        lowered = _lower_feedback_gate(cirq, gate.sub_gate, issues, index)
        if lowered is not None and lowered[0] in (
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "phase",
        ):
            return "c" + lowered[0], lowered[1]
        # Fixed S/T have precisely the same diagonal phase convention.
        if isinstance(gate.sub_gate, cirq.ZPowGate) and gate.sub_gate.global_shift == 0:
            exponent = _real_scalar(
                gate.sub_gate.exponent, issues, index, "controlled phase power"
            )
            if exponent is not None:
                return "cphase", {"theta": pi * exponent}
    return _lower_gate(cirq, gate, issues, index)


class _Feedback:
    """Lower key histories and Boolean predicates to native slot assignments."""

    def __init__(self) -> None:
        self.next_slot = 0
        self.history: dict[str, list[tuple[int, ...]]] = {}
        self.gates: list[Instruction] = []

    def fresh(self) -> int:
        result = self.next_slot
        self.next_slot += 1
        return result

    def word(self, key: str, index: int = -1) -> tuple[int, ...]:
        try:
            return self.history[key][index]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Measurement key {key!r} record {index} is undefined"
            ) from exc

    def xor(self, values: Sequence[Bit], constant: bool = False) -> Bit:
        slots: list[int] = []
        for value in values:
            if type(value) is bool:
                constant ^= value
            else:
                slots.append(value)
        if not slots:
            return constant
        if len(slots) == 1 and not constant:
            return slots[0]
        target = self.fresh()
        self.gates.append(ClassicalXor(tuple(slots), target, constant).instruction())
        return target

    def conjunction(self, values: Sequence[Bit]) -> Bit:
        if any(value is False for value in values):
            return False
        slots = sorted({value for value in values if type(value) is not bool})
        if not slots:
            return True
        if len(slots) == 1:
            return slots[0]
        target = self.fresh()
        self.gates.append(ClassicalXor((), target).instruction())
        assignment = ClassicalXor((), target, True).instruction()
        self.gates.append(
            Instruction(
                assignment.name,
                (),
                assignment.params,
                metadata=dict(assignment.metadata)
                | {"conditions": tuple((slot, 1) for slot in slots)},
            )
        )
        return target

    def disjunction(self, values: Sequence[Bit]) -> Bit:
        if any(value is True for value in values):
            return True
        slots = [value for value in values if value is not False]
        if not slots:
            return False
        if len(slots) == 1:
            return slots[0]
        return self.xor(
            [self.conjunction([self.xor([value], True) for value in slots])], True
        )

    def recorded_word(self, value: Any) -> tuple[int, ...]:
        from sympy import Indexed, Integer, Symbol

        if isinstance(value, Symbol):
            if value != Symbol(value.name):
                raise ValueError("Feedback key symbols must use canonical assumptions")
            return self.word(str(value))
        if isinstance(value, Indexed):
            if len(value.indices) != 1 or not isinstance(value.indices[0], Integer):
                raise ValueError("Feedback bit indexing requires one concrete integer")
            try:
                return (self.recorded_word(value.base.label)[int(value.indices[0])],)
            except IndexError as exc:
                raise ValueError(
                    "Feedback bit index is outside the recorded word"
                ) from exc
        raise ValueError(
            "Feedback equality requires measurement words or integer constants"
        )

    def equality(self, left: Any, right: Any) -> Bit:
        from sympy import Integer

        if isinstance(left, Integer) and isinstance(right, Integer):
            return bool(left == right)
        if isinstance(left, Integer):
            left, right = right, left
        word = self.recorded_word(left)
        if isinstance(right, Integer):
            number = int(right)
            if not 0 <= number < 1 << len(word):
                return False
            return self.conjunction(
                [
                    slot
                    if number >> (len(word) - 1 - i) & 1
                    else self.xor([slot], True)
                    for i, slot in enumerate(word)
                ]
            )
        other = self.recorded_word(right)
        size = max(len(word), len(other))
        padded_left: tuple[Bit, ...] = (False,) * (size - len(word)) + word
        padded_right: tuple[Bit, ...] = (False,) * (size - len(other)) + other
        return self.conjunction(
            [
                self.xor([a, b], True)
                for a, b in zip(padded_left, padded_right, strict=True)
            ]
        )

    def expression(self, expression: Any) -> Bit:
        from sympy import (
            ITE,
            And,
            Eq,
            Equivalent,
            Implies,
            Indexed,
            Integer,
            Ne,
            Not,
            Or,
            Symbol,
            Xor,
            false,
            true,
        )

        if expression == true:
            return True
        if expression == false:
            return False
        if isinstance(expression, Integer):
            return bool(expression)
        if isinstance(expression, (Symbol, Indexed)):
            return self.disjunction(self.recorded_word(expression))
        if isinstance(expression, (Eq, Ne)):
            equal = self.equality(*expression.args)
            return self.xor([equal], True) if isinstance(expression, Ne) else equal
        arguments = [self.expression(arg) for arg in expression.args]
        if isinstance(expression, And):
            return self.conjunction(arguments)
        if isinstance(expression, Or):
            return self.disjunction(arguments)
        if isinstance(expression, Not):
            return self.xor(arguments, True)
        if isinstance(expression, Xor):
            return self.xor(arguments)
        if isinstance(expression, Implies):
            return self.disjunction([self.xor([arguments[0]], True), arguments[1]])
        if isinstance(expression, Equivalent):
            return self.conjunction(
                [self.xor([arguments[0], value], True) for value in arguments[1:]]
            )
        if isinstance(expression, ITE):
            condition, positive, negative = arguments
            return self.disjunction(
                [
                    self.conjunction([condition, positive]),
                    self.conjunction([self.xor([condition], True), negative]),
                ]
            )
        raise ValueError(f"Unsupported feedback expression {expression!r}")

    def condition(self, cirq: Any, condition: Any) -> Bit:
        if isinstance(condition, cirq.KeyCondition):
            return self.disjunction(self.word(str(condition.key), condition.index))
        if isinstance(condition, cirq.SympyCondition):
            return self.expression(condition.expr)
        raise ValueError(f"Unsupported feedback condition {condition!r}")


def import_cirq_feedback(
    circuit: object, *, qubit_order: Sequence[object]
) -> CirqImportResult:
    """Convert finite binary gates, measurements, reset and Boolean feedback.

    Explicit wire ordering includes idle wires and accepts arbitrary binary
    Qids. Measurements may repeat keys or invert recorded bits. Key conditions
    select any prior record and treat a multibit word as a big-endian integer.
    Sympy conditions support Boolean operators and equality to integer words;
    other arithmetic, confusion maps and conditional measurements are rejected.
    Global phase requires at least one declared anchor wire. No lossy mode exists.
    """
    cirq = _cirq()
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError("Feedback import requires cirq.Circuit")
    order = tuple(qubit_order)
    if any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in order) or len(
        set(order)
    ) != len(order):
        raise ValueError("Wire order requires distinct binary Cirq Qids")
    wires = {qubit: index for index, qubit in enumerate(order)}
    if not set(circuit.all_qubits()) <= set(wires):
        raise ValueError("Wire order must include every circuit qubit")
    feedback = _Feedback()
    for index, original in enumerate(circuit.all_operations()):
        issues: list[CirqConversionIssue] = []
        try:
            if original.tags:
                raise ValueError("Tagged operations need explicit native semantics")
            predicate = feedback.conjunction(
                [
                    feedback.condition(cirq, condition)
                    for condition in sorted(original.classical_controls, key=str)
                ]
            )
            operation = original.without_classical_controls()
            gate = operation.gate
            operands = tuple(wires[q] for q in operation.qubits)
            metadata: dict[str, Any] = {}
            if type(predicate) is not bool:
                metadata["conditions"] = ((predicate, 1),)
            emitted: tuple[Instruction, ...]
            if isinstance(gate, cirq.MeasurementGate):
                if original.classical_controls or gate.confusion_map:
                    raise ValueError("Conditional or noisy measurement is unsupported")
                bits: list[int] = []
                for position, wire in enumerate(operands):
                    slot = feedback.fresh()
                    feedback.gates.append(
                        Instruction(
                            "measure",
                            (wire,),
                            metadata={"is_dynamic": True, "classical_bit": slot},
                        )
                    )
                    if position < len(gate.invert_mask) and gate.invert_mask[position]:
                        feedback.gates.append(
                            ClassicalXor((slot,), slot, True).instruction()
                        )
                    bits.append(slot)
                feedback.history.setdefault(
                    cirq.measurement_key_name(operation), []
                ).append(tuple(bits))
                continue
            if isinstance(gate, cirq.ResetChannel) and len(operands) == 1:
                effect = {"is_dynamic": True}
                if metadata:
                    effect["guarded_effect"] = True
                emitted = (Instruction("reset", operands, metadata=effect | metadata),)
            elif isinstance(gate, cirq.GlobalPhaseGate):
                if not order:
                    raise ValueError("Scalar phase import needs a declared anchor wire")
                emitted = global_phase_ir(
                    float(global_phase_exponent(complex(gate.coefficient))), anchor=0
                )
            else:
                lowered = _lower_feedback_gate(cirq, gate, issues, index)
                if lowered is None:
                    raise ValueError(f"Unsupported primitive gate {gate!r}")
                name, params = lowered
                emitted = (Instruction(name, operands, params),)
            if predicate is not False:
                feedback.gates.extend(
                    Instruction(
                        g.name, g.wires, g.params, metadata=dict(g.metadata) | metadata
                    )
                    for g in emitted
                )
        except (ValueError, TypeError) as exc:
            issues.append(
                CirqConversionIssue(
                    "feedback_import_not_representable", str(exc), "error", index
                )
            )
            report = _report("from_cirq", str(cirq.__version__), issues)
            raise CirqConversionError(str(exc), report) from exc
    instructions = tuple(feedback.gates)
    validate_classical_control(instructions, n_qubits=len(order))
    report = _report("from_cirq", str(cirq.__version__), [])
    ir = CircuitIR(
        len(order),
        instructions,
        dtype="complex128",
        metadata={
            "interop": {
                "source": "cirq",
                "conversion_report": report.to_dict(),
                "measurement_records": {
                    key: tuple(records) for key, records in feedback.history.items()
                },
            }
        },
    ).validate()
    return CirqImportResult(ir, report)
