"""Compile canonical FlagQuantum IR to QIR base-profile LLVM IR text.

QIR is an open specification maintained by the QIR Alliance, so this emitter
depends on no vendor component. It declares the base-profile entry point, the
``__quantum__qis__*`` calls for the lowered program, and the ``__quantum__rt__*``
calls that record the program output.

Only the constant-index pointer kind is emitted: a qubit or result operand is
``ptr null`` for index 0 and ``ptr inttoptr (i64 k to ptr)`` otherwise, and both
dynamic-management module flags are ``false``. Profile-QIR, the adaptive
profile, pulse-level generation, and runtime library extension calls are out of
scope.

Opcodes without a QIS instruction are lowered to a native sequence whose unitary
equals the requested gate up to an unconditional global phase. The base profile
cannot express that phase and recorded measurement output cannot observe it, so
the emitted program is equivalent for every measurable outcome. Conformance is
therefore checked with a global-phase-invariant overlap instead of an
elementwise comparison. A gate whose semantics the base profile cannot carry at
all is refused rather than approximated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.parameters import is_parameterized_value, parameter_names_in_value
from .operator_lowering import require_static_gate_program, validate_lowering

ENTRY_POINT_NAME = "FlagQuantumEntryPoint"
OUTPUT_LABELING_SCHEMA = "labeled"
QIR_MAJOR_VERSION = 2
QIR_MINOR_VERSION = 0

# QIS names declared by the QIR instruction set. `cx` is spelled `cnot`, the
# adjoint gates use the `__adj` suffix, and `mz` is irreversible.
_QIS_SIGNATURES = {
    "h__body": "(ptr)",
    "s__body": "(ptr)",
    "s__adj": "(ptr)",
    "t__body": "(ptr)",
    "t__adj": "(ptr)",
    "x__body": "(ptr)",
    "y__body": "(ptr)",
    "z__body": "(ptr)",
    "rz__body": "(double, ptr)",
    "rx__body": "(double, ptr)",
    "ry__body": "(double, ptr)",
    "cnot__body": "(ptr, ptr)",
    "cz__body": "(ptr, ptr)",
    "swap__body": "(ptr, ptr)",
    "ccx__body": "(ptr, ptr, ptr)",
    "mz__body": "(ptr, ptr writeonly) #1",
}
_NATIVE_QIS = {
    "h": "h__body",
    "s": "s__body",
    "sdg": "s__adj",
    "t": "t__body",
    "tdg": "t__adj",
    "x": "x__body",
    "y": "y__body",
    "z": "z__body",
    "cx": "cnot__body",
    "cz": "cz__body",
    "swap": "swap__body",
    "ccx": "ccx__body",
}
_ROTATION_QIS = {"rx": "rx__body", "ry": "ry__body", "rz": "rz__body"}
_RUNTIME_SIGNATURES = {
    "__quantum__rt__initialize": "(ptr)",
    "__quantum__rt__tuple_record_output": "(i64, ptr)",
    "__quantum__rt__result_record_output": "(ptr, ptr)",
}


@dataclass(frozen=True)
class _QisCall:
    """One ``__quantum__qis__`` invocation with operands already resolved."""

    name: str
    qubits: tuple[int, ...] = ()
    angle: float | None = None


def _one(name: str, wire: int, angle: float | None = None) -> _QisCall:
    return _QisCall(name, (wire,), angle)


def _cnot(control: int, target: int) -> _QisCall:
    return _QisCall("cnot__body", (control, target))


def _rz(wire: int, theta: float) -> _QisCall:
    return _QisCall("rz__body", (wire,), theta)


def _rzz(control: int, target: int, theta: float) -> tuple[_QisCall, ...]:
    return (_cnot(control, target), _rz(target, theta), _cnot(control, target))


def _crz(control: int, target: int, theta: float) -> tuple[_QisCall, ...]:
    return (
        _rz(target, theta / 2),
        _cnot(control, target),
        _rz(target, -theta / 2),
        _cnot(control, target),
    )


def _double_literal(number: float) -> str:
    """Return the exact LLVM ``double`` literal for a binary64 value.

    LLVM requires a decimal point or an exponent separator, so ``repr`` and
    ``float.hex`` are both rejected by the LLVM parser. Seventeen significant
    digits round-trip every binary64 value exactly.
    """

    return f"{number:.17e}"


def _parameter(instruction: Instruction, name: str) -> float:
    schema = get_operator_schema(instruction.name)
    if schema is None:
        raise ValueError(f"QIR cannot emit unknown gate {instruction.name!r}.")
    if name not in schema.parameters or name not in instruction.params:
        raise ValueError(f"QIR gate {schema.opcode!r} is missing parameter {name!r}.")
    value: Any = instruction.params[name]
    if is_parameterized_value(value):
        names = ", ".join(parameter_names_in_value(value))
        raise ValueError(
            f"Unbound circuit parameter(s): {names}. Call bind_parameters first."
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("QIR emission requires scalar gate parameters.")
        value = value.detach().reshape(()).cpu().item()
    if not isinstance(value, Real):
        raise TypeError(
            f"QIR gate {schema.opcode!r} parameter {name!r} must be real, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"QIR gate {schema.opcode!r} parameter {name!r} must be finite."
        )
    return number


def _calls(instruction: Instruction) -> tuple[_QisCall, ...]:
    """Return the native QIS sequence emitted for one instruction."""

    opcode = canonical_opcode(instruction.name)
    wires = instruction.wires
    if opcode == "i":
        return ()
    if opcode in _NATIVE_QIS:
        return (_QisCall(_NATIVE_QIS[opcode], wires),)
    if opcode in _ROTATION_QIS:
        return (
            _QisCall(_ROTATION_QIS[opcode], wires, _parameter(instruction, "theta")),
        )
    if opcode in {"phase", "u1"}:
        return (_rz(wires[0], _parameter(instruction, "theta")),)
    if opcode == "sx":
        wire = wires[0]
        return (_one("h__body", wire), _one("s__body", wire), _one("h__body", wire))
    if opcode == "sxdg":
        wire = wires[0]
        return (_one("h__body", wire), _one("s__adj", wire), _one("h__body", wire))
    if opcode == "u3":
        return (
            _rz(wires[0], _parameter(instruction, "lbd")),
            _one("ry__body", wires[0], _parameter(instruction, "theta")),
            _rz(wires[0], _parameter(instruction, "phi")),
        )
    if opcode == "u2":
        return (
            _rz(wires[0], _parameter(instruction, "lbd")),
            _one("ry__body", wires[0], math.pi / 2),
            _rz(wires[0], _parameter(instruction, "phi")),
        )
    if opcode == "cy":
        control, target = wires
        return (
            _one("s__adj", target),
            _cnot(control, target),
            _one("s__body", target),
        )
    if opcode == "cswap":
        first, second, third = wires
        return (
            _cnot(second, third),
            _QisCall("ccx__body", (first, third, second)),
            _cnot(second, third),
        )
    if opcode == "rzz":
        return _rzz(wires[0], wires[1], _parameter(instruction, "theta"))
    if opcode == "rxx":
        left, right = wires
        return (
            _one("h__body", left),
            _one("h__body", right),
            *_rzz(left, right, _parameter(instruction, "theta")),
            _one("h__body", left),
            _one("h__body", right),
        )
    if opcode == "ryy":
        left, right = wires
        return (
            _one("s__adj", left),
            _one("s__adj", right),
            _one("h__body", left),
            _one("h__body", right),
            *_rzz(left, right, _parameter(instruction, "theta")),
            _one("h__body", left),
            _one("h__body", right),
            _one("s__body", left),
            _one("s__body", right),
        )
    if opcode == "crz":
        return _crz(wires[0], wires[1], _parameter(instruction, "theta"))
    if opcode == "crx":
        control, target = wires
        return (
            _one("h__body", target),
            *_crz(control, target, _parameter(instruction, "theta")),
            _one("h__body", target),
        )
    if opcode == "cry":
        control, target = wires
        return (
            _one("s__adj", target),
            _one("h__body", target),
            *_crz(control, target, _parameter(instruction, "theta")),
            _one("h__body", target),
            _one("s__body", target),
        )
    if opcode == "cphase":
        control, target = wires
        theta = _parameter(instruction, "theta")
        return (
            _rz(control, theta / 2),
            _rz(target, theta / 2),
            *_rzz(control, target, -theta / 2),
        )
    raise ValueError(f"QIR base profile has no lowering for gate {opcode!r}.")


def _qubit_operand(index: int, *, writeonly: bool = False) -> str:
    if index == 0:
        return "ptr writeonly null" if writeonly else "ptr null"
    if writeonly:
        return f"ptr writeonly inttoptr (i64 {index} to ptr)"
    return f"ptr inttoptr (i64 {index} to ptr)"


def _label_constant(label: str) -> str:
    return f"[{len(label.encode('utf-8')) + 1} x i8] c\"{label}\\00\""


def _resolved_result_wires(
    ir: CircuitIR, result_wires: tuple[int, ...] | None
) -> tuple[int, ...]:
    if result_wires is None:
        if len(ir.measurements) > 1:
            raise ValueError(
                "QIR base profile requires at most one terminal measurement."
            )
        if not ir.measurements:
            return ()
        measurement = ir.measurements[0]
        if measurement.kind != "samples":
            raise ValueError(
                f"QIR base profile cannot record a {measurement.kind!r} measurement."
            )
        wires = tuple(measurement.wires)
    else:
        wires = tuple(result_wires)
    if (
        not wires
        or len(set(wires)) != len(wires)
        or any(
            type(wire) is not int or wire < 0 or wire >= ir.n_wires for wire in wires
        )
    ):
        raise ValueError("QIR result wires must be unique in-range integers.")
    return wires


def _validated_ir(program: Any) -> CircuitIR:
    ir = ensure_circuit_ir(program)
    validate_lowering(ir, "qir")
    return require_static_gate_program(ir, language="QIR base profile")


def _call_text(call: _QisCall) -> str:
    operands = [_qubit_operand(wire) for wire in call.qubits]
    if call.angle is not None:
        operands.insert(0, f"double {_double_literal(float(call.angle))}")
    return f"call void @__quantum__qis__{call.name}({', '.join(operands)})"


def emit_qir(program: Any, *, result_wires: tuple[int, ...] | None = None) -> str:
    """Return deterministic QIR base-profile LLVM IR text.

    Args:
        program: Canonical ``CircuitIR``, or an object exposing ``to_ir()``.
        result_wires: Physical wires measured into consecutive result values.
            Defaults to the wires of the single terminal samples measurement, or
            to no measurement when the program declares none.

    Raises:
        ValueError: The program cannot be represented by the QIR base profile,
            or a gate parameter is unbound or is not a finite real number.
    """

    ir = _validated_ir(program)
    measured = _resolved_result_wires(ir, result_wires)
    body: list[_QisCall] = []
    for instruction in ir.instructions:
        body.extend(_calls(instruction))

    labels = ("t0", *(f"r{index}" for index in range(len(measured))))
    constants = {label: f"@{index}" for index, label in enumerate(labels)}

    lines = ["; FlagQuantum QIR base profile module", ""]
    lines.extend(
        f"{constants[label]} = internal constant {_label_constant(label)}"
        for label in labels
    )
    lines.extend(["", f"define i64 @{ENTRY_POINT_NAME}() #0 {{", "entry:"])
    lines.append("  call void @__quantum__rt__initialize(ptr null)")
    lines.extend(["  br label %body", "", "body:"])
    lines.extend(f"  {_call_text(call)}" for call in body)
    lines.extend(["  br label %measurements", "", "measurements:"])
    lines.extend(
        "  call void @__quantum__qis__mz__body"
        f"({_qubit_operand(wire)}, {_qubit_operand(result, writeonly=True)})"
        for result, wire in enumerate(measured)
    )
    lines.extend(["  br label %output", "", "output:"])
    lines.append(
        "  call void @__quantum__rt__tuple_record_output"
        f"(i64 {len(measured)}, ptr {constants['t0']})"
    )
    lines.extend(
        "  call void @__quantum__rt__result_record_output"
        f"({_qubit_operand(result)}, ptr {constants[f'r{result}']})"
        for result in range(len(measured))
    )
    lines.extend(["  ret i64 0", "}", ""])

    declarations = {f"__quantum__qis__{call.name}" for call in body}
    if measured:
        declarations.add("__quantum__qis__mz__body")
    signatures = dict(_RUNTIME_SIGNATURES)
    for name in declarations:
        signatures[name] = _QIS_SIGNATURES[name.removeprefix("__quantum__qis__")]
    if not measured:
        del signatures["__quantum__rt__result_record_output"]
    lines.extend(
        f"declare void @{name}{signatures[name]}" for name in sorted(signatures)
    )
    lines.extend(
        [
            "",
            'attributes #0 = { "entry_point" "qir_profiles"="base_profile" '
            f'"output_labeling_schema"="{OUTPUT_LABELING_SCHEMA}" '
            f'"required_num_qubits"="{ir.n_wires}" '
            f'"required_num_results"="{len(measured)}" }}',
        ]
    )
    if measured:
        lines.extend(["", 'attributes #1 = { "irreversible" }'])
    lines.extend(
        [
            "",
            "!llvm.module.flags = !{!0, !1, !2, !3}",
            "",
            f'!0 = !{{i32 1, !"qir_major_version", i32 {QIR_MAJOR_VERSION}}}',
            f'!1 = !{{i32 7, !"qir_minor_version", i32 {QIR_MINOR_VERSION}}}',
            '!2 = !{i32 1, !"dynamic_qubit_management", i1 false}',
            '!3 = !{i32 1, !"dynamic_result_management", i1 false}',
        ]
    )
    return "\n".join(lines)


__all__ = ("ENTRY_POINT_NAME", "OUTPUT_LABELING_SCHEMA", "emit_qir")
