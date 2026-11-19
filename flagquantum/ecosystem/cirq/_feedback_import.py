"""Explicit lossless import of measured Cirq circuits and Boolean feedback.

External measurement keys end here. Native slots represent each recorded bit
and any composed Boolean predicate. The existing static conversion is separate.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from math import pi
from typing import Any

from ...compiler.registers.primitives.global_phase import (
    scalar_phase_ir,
)
from ...core.classical_control import validate_classical_control
from ...core.classical_dataflow import ClassicalXor
from ...core.ir import CircuitIR, Instruction
from ._channel import _channel_instruction
from .conversion import (
    _check_cirq_symbol_identity,
    _cirq,
    _eigen_phase_operations,
    _flagquantum_parameter_value,
    _global_phase_angle,
    _lower_gate,
    _real_scalar,
    _report,
)
from .models import CirqConversionError, CirqConversionIssue, CirqImportResult

Bit = int | bool


def _scalar_control_operations(cirq: Any, operations: Iterable[Any]) -> Iterator[Any]:
    """Express negative phase and rotation controls in the positive basis."""
    for original in operations:
        if original.tags:
            yield original
            continue
        operation = original.without_classical_controls()
        gate = operation.gate
        if isinstance(gate, cirq.ControlledGate) and isinstance(
            gate.sub_gate,
            (cirq.GlobalPhaseGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate),
        ):
            values = tuple(tuple(value) for value in gate.control_values)
            if (
                len(values) <= 2
                and all(value in ((0,), (1,)) for value in values)
                and (0,) in values
            ):
                flips = tuple(
                    cirq.X(qubit)
                    for qubit, value in zip(
                        operation.qubits[: len(values)], values, strict=True
                    )
                    if value == (0,)
                )
                positive = gate.sub_gate.controlled(num_controls=len(values)).on(
                    *operation.qubits
                )
                for leaf in (*flips, positive, *reversed(flips)):
                    yield leaf.with_classical_controls(*original.classical_controls)
                continue
        if (
            isinstance(gate, cirq.ControlledGate)
            and gate.num_controls() == 2
            and isinstance(gate.sub_gate, cirq.YPowGate)
        ):
            # Cirq 1.7 CCYPowGate uses this same S-adjoint/CCX/S basis change.
            # Apply it directly so Cirq 1.6 need not resolve a symbolic matrix.
            target = operation.qubits[-1]
            child = gate.sub_gate
            equivalent = (
                cirq.XPowGate(exponent=child.exponent, global_shift=child.global_shift)
                .controlled(num_controls=2, control_values=gate.control_values)
                .on(*operation.qubits)
            )
            for leaf in (cirq.S(target) ** -1, equivalent, cirq.S(target)):
                yield leaf.with_classical_controls(*original.classical_controls)
            continue
        yield original


def _lower_feedback_gate(
    cirq: Any, gate: Any, issues: list[CirqConversionIssue], index: int
) -> tuple[str, dict[str, Any]] | None:
    """Retain exact phase powers and canonical singly controlled primitives."""
    if isinstance(gate, cirq.ControlledGate) and isinstance(
        gate.sub_gate, cirq.GlobalPhaseGate
    ):
        values = tuple(tuple(value) for value in gate.control_values)
        if values in (((1,),), ((1,), (1,))):
            return ("phase" if len(values) == 1 else "cphase"), {
                "theta": _global_phase_angle(cirq, gate.sub_gate.coefficient)
            }
    if gate == cirq.Y.controlled():
        return "cy", {}
    if gate in (cirq.ISWAP, cirq.ISWAP**-1):
        return ("iswap" if gate == cirq.ISWAP else "iswapdg"), {}
    if isinstance(gate, (cirq.XPowGate, cirq.CXPowGate)) and gate.global_shift == 0:
        exponent = _real_scalar(gate.exponent, issues, index, "X power")
        if exponent is not None and exponent % 2 in (0.5, 1.5):
            name = "sx" if exponent % 2 == 0.5 else "sxdg"
            return ("c" + name if isinstance(gate, cirq.CXPowGate) else name), {}
    if isinstance(gate, (cirq.ZPowGate, cirq.CZPowGate)) and gate.global_shift == 0:
        if cirq.is_parameterized(gate.exponent):
            try:
                theta = _flagquantum_parameter_value(
                    cirq, gate.exponent, scale_by_pi=True
                )
            except (TypeError, ValueError):
                return _lower_gate(cirq, gate, issues, index)
            return ("cphase" if isinstance(gate, cirq.CZPowGate) else "phase"), {
                "theta": theta
            }
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
            "sx",
            "sxdg",
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


def _has_explicit_matrix(cirq: Any, gate: Any) -> bool:
    """Recognize declared matrix leaves without contracting arbitrary components."""
    if type(gate) in (cirq.XPowGate, cirq.YPowGate) and (
        _lower_feedback_gate(cirq, gate, [], 0) is None
    ):
        # These two source rotation atoms expose a 2x2 unitary while their
        # decomposition hook returns NotImplemented for generic powers.
        return True
    while True:
        if isinstance(gate, cirq.ControlledGate):
            gate = gate.sub_gate
        elif (
            getattr(gate, "ctrl_spec", None) is not None
            and callable(getattr(gate, "_unitary_", None))
            and getattr(gate, "subbloq", None) is not None
        ):
            # Source control wrappers retain an explicit matrix leaf and expose
            # its actual controlled unitary protocol. Do not synthesize it.
            gate = gate.subbloq
        else:
            break
    if isinstance(gate, cirq.MatrixGate) or (
        getattr(gate, "matrix", None) is not None
        and callable(getattr(gate, "_unitary_", None))
    ):
        return True
    # A pure Cirq leaf may supply its matrix only through the unitary protocol.
    # Keep provider decompositions and register-component boundaries intact;
    # their unitary protocols may contract an entire algorithm.
    return (
        isinstance(gate, cirq.Gate)
        # Scalar phases have their own importer and controlled decomposition.
        # Unwrapping a control must not turn that scalar into a matrix atom.
        and not isinstance(gate, cirq.GlobalPhaseGate)
        and not hasattr(gate, "signature")
        and callable(getattr(gate, "_unitary_", None))
        and not callable(getattr(gate, "_decompose_", None))
        and not callable(getattr(gate, "_decompose_with_context_", None))
        and _lower_feedback_gate(cirq, gate, [], 0) is None
    )


def _matrix_instruction(cirq: Any, gate: Any, operands: tuple[int, ...]) -> Instruction:
    import numpy as np
    import torch

    matrix = np.array(cirq.unitary(gate), dtype=np.complex128, copy=True)
    size = 1 << len(operands)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(
            "Explicit unitary matrix must be finite and match its operands"
        )
    if not np.allclose(matrix.conj().T @ matrix, np.eye(size), atol=1e-10, rtol=0):
        raise ValueError("Explicit matrix leaf is not unitary")
    return Instruction("any", operands, matrix=torch.from_numpy(matrix))


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
                    (
                        slot
                        if number >> (len(word) - 1 - i) & 1
                        else self.xor([slot], True)
                    )
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
    Global phase requires at least one declared anchor wire. Explicit unitary
    matrix leaves retain copied dense operands; no synthesis is inferred.
    No lossy mode exists.
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
    _check_cirq_symbol_identity(cirq, circuit.all_operations())
    feedback = _Feedback()
    for index, original in enumerate(
        _scalar_control_operations(
            cirq, _eigen_phase_operations(cirq, circuit.all_operations())
        )
    ):
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
                emitted = scalar_phase_ir(
                    0, _global_phase_angle(cirq, gate.coefficient)
                )
            elif _has_explicit_matrix(cirq, gate):
                emitted = (_matrix_instruction(cirq, gate, operands),)
            elif (channel := _channel_instruction(gate, operands)) is not None:
                emitted = (channel,)
            else:
                lowered = _lower_feedback_gate(cirq, gate, issues, index)
                if lowered is None:
                    raise ValueError(f"Unsupported primitive gate {gate!r}")
                name, params = lowered
                emitted = (Instruction(name, operands, params),)
            if predicate is not False:
                feedback.gates.extend(
                    Instruction(
                        g.name,
                        g.wires,
                        g.params,
                        matrix=g.matrix,
                        metadata=dict(g.metadata) | metadata,
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
