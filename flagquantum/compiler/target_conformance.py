"""Strict conformance checks for Compiler-emitted target text."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace

from ..core.ir import CircuitIR, Instruction, MeasurementNode
from ..errors import CompilationError
from .openqasm_gates import FIXED_GATES, PARAMETERIZED_GATES
from .target_emission import TargetEmissionResult, emit_legalized_target
from .target_legalization import TargetLegalizationResult


class TargetConformanceError(CompilationError):
    """Emitted target text or its evidence fails strict conformance."""


@dataclass(frozen=True)
class TargetConformanceResult:
    """Private parsed-program evidence, not an execution certification."""

    emission_identity: str
    reconstructed_program: CircuitIR
    reconstructed_circuit_hash: str
    parsed_operation_count: int
    conformance_identity: str


_QASM_GATE = re.compile(
    r"(?:(pow\(-1\) @) )?([A-Za-z][A-Za-z0-9_]*)"
    r"(?:\(([^()]*)\))? "
    r"(q\[[0-9]+\](?:, q\[[0-9]+\])*)\;"
)
_QASM_WIRE = re.compile(r"q\[([0-9]+)\]")
_QASM2_QREG = re.compile(r"qreg q\[([1-9][0-9]*)\];")
_QASM2_CREG = re.compile(r"creg c\[([1-9][0-9]*)\];")
_QASM3_QREG = re.compile(r"qubit\[([1-9][0-9]*)\] q;")
_QASM3_CREG = re.compile(r"bit\[([1-9][0-9]*)\] c;")

# The strict round-trip check recognises exactly the spellings import recognises,
# so both directions read one table rather than two copies of it.
_FIXED_QASM_GATES = FIXED_GATES
_PARAMETERIZED_QASM_GATES = PARAMETERIZED_GATES


def _reconstructed_program(
    template: CircuitIR,
    instructions: tuple[Instruction, ...],
) -> CircuitIR:
    result_wires = _allocated_result_wires(template)
    if result_wires is None:
        measurement = MeasurementNode(
            "samples", tuple(range(template.n_wires)), shots=None
        )
    else:
        measurement = MeasurementNode(
            "samples",
            result_wires,
            shots=None,
            metadata={
                "fq_logical_wires": tuple(range(len(result_wires))),
                "fq_result_order": "logical_wire_order",
            },
        )
    return replace(
        template,
        instructions=instructions,
        observables=(),
        measurements=(measurement,),
    )


def _allocated_result_wires(template: CircuitIR) -> tuple[int, ...] | None:
    routing = template.metadata.get("routing")
    if not isinstance(routing, dict) or routing.get("schema") != (
        "flagquantum_directed_routing_plan_v2"
    ):
        return None
    wires = routing.get("logical_result_physical_slots")
    if not isinstance(wires, tuple) or not wires:
        raise TargetConformanceError("allocated template lacks logical result slots")
    return wires


def _finite_numbers(text: str | None, *, owner: str) -> tuple[float, ...]:
    if text is None:
        return ()
    raw = text.split(",")
    if any(not item.strip() for item in raw):
        raise TargetConformanceError(f"{owner} has an empty parameter")
    try:
        values = tuple(float(item.strip()) for item in raw)
    except ValueError as error:
        raise TargetConformanceError(f"{owner} has a non-numeric parameter") from error
    if any(not math.isfinite(value) for value in values):
        raise TargetConformanceError(f"{owner} has a non-finite parameter")
    return values


def _parse_qasm_instruction(line: str, *, n_wires: int, index: int) -> Instruction:
    match = _QASM_GATE.fullmatch(line)
    if match is None:
        raise TargetConformanceError(f"OpenQASM body line {index} is not canonical")
    inverse, raw_name, parameter_text, operand_text = match.groups()
    wires = tuple(int(item) for item in _QASM_WIRE.findall(operand_text))
    if any(wire >= n_wires for wire in wires):
        raise TargetConformanceError(
            f"OpenQASM body line {index} references an out-of-range wire"
        )
    values = _finite_numbers(parameter_text, owner=f"OpenQASM body line {index}")

    if inverse is not None:
        if raw_name != "sx" or values:
            raise TargetConformanceError(
                f"OpenQASM body line {index} has an unsupported inverse modifier"
            )
        return Instruction("sxdg", wires)
    parameter_names: tuple[str, ...]
    if raw_name == "U":
        opcode, parameter_names = "u3", ("theta", "phi", "lbd")
    elif raw_name in _FIXED_QASM_GATES:
        if values:
            raise TargetConformanceError(
                f"OpenQASM body line {index} gives parameters to a fixed gate"
            )
        return Instruction(_FIXED_QASM_GATES[raw_name], wires)
    else:
        try:
            opcode, parameter_names = _PARAMETERIZED_QASM_GATES[raw_name]
        except KeyError as error:
            raise TargetConformanceError(
                f"OpenQASM body line {index} uses unsupported gate {raw_name!r}"
            ) from error
    if len(values) != len(parameter_names):
        raise TargetConformanceError(
            f"OpenQASM body line {index} has the wrong parameter count"
        )
    return Instruction(
        opcode, wires, params=dict(zip(parameter_names, values, strict=True))
    )


def _parse_openqasm(
    text: str,
    *,
    profile: str,
    template: CircuitIR,
) -> CircuitIR:
    lines = text.splitlines()
    if profile == "openqasm-2.0":
        prefix = ("OPENQASM 2.0;", 'include "qelib1.inc";')
        qreg_pattern, creg_pattern = _QASM2_QREG, _QASM2_CREG
    else:
        prefix = ("OPENQASM 3.0;", 'include "stdgates.inc";')
        qreg_pattern, creg_pattern = _QASM3_QREG, _QASM3_CREG
    if len(lines) < 5 or tuple(lines[:2]) != prefix:
        raise TargetConformanceError(f"{profile} header is not canonical")
    qreg = qreg_pattern.fullmatch(lines[2])
    creg = creg_pattern.fullmatch(lines[3])
    result_wires = _allocated_result_wires(template)
    expected_result_width = (
        template.n_wires if result_wires is None else len(result_wires)
    )
    if qreg is None or creg is None or int(creg.group(1)) != expected_result_width:
        raise TargetConformanceError(f"{profile} register declarations are invalid")
    n_wires = int(qreg.group(1))
    if n_wires != template.n_wires:
        raise TargetConformanceError(
            f"{profile} register width does not match the legalized circuit"
        )

    if profile == "openqasm-2.0":
        measured = tuple(range(n_wires)) if result_wires is None else result_wires
        terminal = [
            f"measure q[{wire}] -> c[{result}];" for result, wire in enumerate(measured)
        ]
        if len(lines) < 4 + len(terminal) or lines[-len(terminal) :] != terminal:
            raise TargetConformanceError(
                "openqasm-2.0 terminal measurement is not canonical"
            )
        body = lines[4 : -len(terminal)]
    elif result_wires is None:
        if lines[-1] != "c = measure q;":
            raise TargetConformanceError(
                "openqasm-3.0 terminal measurement is not canonical"
            )
        body = lines[4:-1]
    else:
        terminal = [
            f"c[{result}] = measure q[{wire}];"
            for result, wire in enumerate(result_wires)
        ]
        if len(lines) < 4 + len(terminal) or lines[-len(terminal) :] != terminal:
            raise TargetConformanceError(
                "openqasm-3.0 projected terminal measurement is not canonical"
            )
        body = lines[4 : -len(terminal)]
    instructions = tuple(
        _parse_qasm_instruction(line, n_wires=n_wires, index=index)
        for index, line in enumerate(body)
    )
    return _reconstructed_program(template, instructions)


def _qcis_wire(token: str, *, n_wires: int, index: int) -> int:
    if not re.fullmatch(r"Q[0-9]+", token):
        raise TargetConformanceError(f"QCIS line {index} has an invalid wire")
    wire = int(token[1:])
    if wire >= n_wires:
        raise TargetConformanceError(f"QCIS line {index} has an out-of-range wire")
    return wire


def _parse_qcis(text: str, *, template: CircuitIR) -> CircuitIR:
    instructions: list[Instruction] = []
    for index, line in enumerate(text.splitlines()):
        parts = line.split()
        if len(parts) < 2:
            raise TargetConformanceError(f"QCIS line {index} is not canonical")
        name = parts[0]
        if name in {"X2P", "X2M", "Y2P", "Y2M"}:
            if len(parts) != 2:
                raise TargetConformanceError(f"QCIS line {index} has extra operands")
            wire = _qcis_wire(parts[1], n_wires=template.n_wires, index=index)
            opcode = "rx" if name.startswith("X") else "ry"
            sign = 1.0 if name.endswith("P") else -1.0
            instructions.append(
                Instruction(opcode, (wire,), params={"theta": sign * math.pi / 2})
            )
        elif name == "RZ":
            if len(parts) != 3:
                raise TargetConformanceError(f"QCIS line {index} is not canonical RZ")
            wire = _qcis_wire(parts[1], n_wires=template.n_wires, index=index)
            (theta,) = _finite_numbers(parts[2], owner=f"QCIS line {index}")
            instructions.append(Instruction("rz", (wire,), params={"theta": theta}))
        elif name == "CZ":
            if len(parts) != 3:
                raise TargetConformanceError(f"QCIS line {index} is not canonical CZ")
            wires = tuple(
                _qcis_wire(item, n_wires=template.n_wires, index=index)
                for item in parts[1:]
            )
            instructions.append(Instruction("cz", wires))
        elif name == "I":
            if len(parts) != 3 or parts[2] != "60":
                raise TargetConformanceError(f"QCIS line {index} is not canonical I")
            wire = _qcis_wire(parts[1], n_wires=template.n_wires, index=index)
            instructions.append(Instruction("i", (wire,)))
        else:
            raise TargetConformanceError(
                f"QCIS line {index} uses unsupported instruction {name!r}"
            )
    return _reconstructed_program(template, tuple(instructions))


_QIR_PREAMBLE = "; FlagQuantum QIR base profile module"
_QIR_LABEL = re.compile(
    r'@([0-9]+) = internal constant \[([0-9]+) x i8\] c"([^"\\]*)\\00"'
)
_QIR_DEFINE = re.compile(r"define i64 @([A-Za-z_][A-Za-z0-9_.]*)\(\) #([0-9]+) \{$")
_QIR_BLOCK = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*):")
_QIR_BRANCH = re.compile(r"br label %([A-Za-z_][A-Za-z0-9_.]*)")
_QIR_CALL = re.compile(r"call void @(__quantum__(?:qis|rt)__[A-Za-z0-9_]+)\((.*)\)$")
_QIR_DECLARE = re.compile(
    r"declare void @(__quantum__(?:qis|rt)__[A-Za-z0-9_]+)(\(.*\))(?: #([0-9]+))?$"
)
_QIR_ATTRIBUTE_GROUP = re.compile(r"attributes #([0-9]+) = \{ (.*) \}$")
_QIR_ATTRIBUTE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"(?:="([^"]*)")?')
_QIR_MODULE_FLAG = re.compile(
    r'!([0-9]+) = !\{i32 ([0-9]+), !"([A-Za-z_]+)", (i32 ([0-9]+)|i1 (true|false))\}$'
)
_QIR_DOUBLE = re.compile(r"double (-?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?)$")
_QIR_POINTER = re.compile(r"ptr( writeonly)? (null|inttoptr \(i64 ([0-9]+) to ptr\))$")
_QIR_NATIVE_GATES = {
    "h__body": ("h", 1),
    "s__body": ("s", 1),
    "s__adj": ("sdg", 1),
    "t__body": ("t", 1),
    "t__adj": ("tdg", 1),
    "x__body": ("x", 1),
    "y__body": ("y", 1),
    "z__body": ("z", 1),
    "cnot__body": ("cx", 2),
    "cz__body": ("cz", 2),
    "swap__body": ("swap", 2),
    "ccx__body": ("ccx", 3),
}
_QIR_ROTATIONS = {"rz__body": "rz", "rx__body": "rx", "ry__body": "ry"}
_QIR_SIGNATURES = {
    "__quantum__qis__h__body": "(ptr)",
    "__quantum__qis__s__body": "(ptr)",
    "__quantum__qis__s__adj": "(ptr)",
    "__quantum__qis__t__body": "(ptr)",
    "__quantum__qis__t__adj": "(ptr)",
    "__quantum__qis__x__body": "(ptr)",
    "__quantum__qis__y__body": "(ptr)",
    "__quantum__qis__z__body": "(ptr)",
    "__quantum__qis__rx__body": "(double, ptr)",
    "__quantum__qis__ry__body": "(double, ptr)",
    "__quantum__qis__rz__body": "(double, ptr)",
    "__quantum__qis__cnot__body": "(ptr, ptr)",
    "__quantum__qis__cz__body": "(ptr, ptr)",
    "__quantum__qis__swap__body": "(ptr, ptr)",
    "__quantum__qis__ccx__body": "(ptr, ptr, ptr)",
    "__quantum__qis__mz__body": "(ptr, ptr writeonly)",
    "__quantum__rt__initialize": "(ptr)",
    "__quantum__rt__tuple_record_output": "(i64, ptr)",
    "__quantum__rt__result_record_output": "(ptr, ptr)",
}
_QIR_IRREVERSIBLE_CALL = "__quantum__qis__mz__body"
_QIR_REQUIRED_FLAGS = {
    "qir_major_version": ("1", "2"),
    "qir_minor_version": ("7", "0"),
    "dynamic_qubit_management": ("1", "false"),
    "dynamic_result_management": ("1", "false"),
}


def _qir_operand(token: str, *, owner: str, writeonly: bool) -> int:
    match = _QIR_POINTER.fullmatch(token)
    if match is None:
        raise TargetConformanceError(f"{owner} has an unsupported pointer operand")
    if (match.group(1) is not None) != writeonly:
        raise TargetConformanceError(f"{owner} has a wrongly qualified result operand")
    return 0 if match.group(2) == "null" else int(match.group(3))


def _qir_mz_operands(line: str, *, owner: str) -> tuple[int, int]:
    match = _QIR_CALL.fullmatch(line)
    assert match is not None
    operands = match.group(2).split(", ")
    if len(operands) != 2:
        raise TargetConformanceError(f"{owner} is not a canonical mz call")
    qubit = _qir_operand(operands[0], owner=owner, writeonly=False)
    result = _qir_operand(operands[1], owner=owner, writeonly=True)
    return qubit, result


def _qir_record_operands(
    line: str,
) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    match = _QIR_CALL.fullmatch(line)
    assert match is not None
    callee = match.group(1).removeprefix("__quantum__rt__")
    if callee == "tuple_record_output":
        count, label = match.group(2).split(", ")
        if not count.startswith("i64 "):
            raise TargetConformanceError("qir-2.0 tuple count is not an i64 constant")
        try:
            width = int(count[4:])
        except ValueError as error:
            raise TargetConformanceError(
                "qir-2.0 tuple count is not an integer constant"
            ) from error
        return callee, (width,), (label,)
    if callee == "result_record_output":
        pointer, label = match.group(2).split(", ")
        return (
            callee,
            (_qir_operand(pointer, owner="qir-2.0 output", writeonly=False),),
            (label,),
        )
    raise TargetConformanceError(f"qir-2.0 has an unsupported runtime call {callee!r}")


def _qir_label_operand(token: str, *, labels: dict[str, str], owner: str) -> str:
    if not token.startswith("ptr @"):
        raise TargetConformanceError(f"{owner} label operand is not a global constant")
    name = token[4:]
    try:
        return labels[name]
    except KeyError as error:
        raise TargetConformanceError(
            f"{owner} references an undeclared label"
        ) from error


def _qir_body_instruction(line: str, *, index: int, n_wires: int) -> Instruction:
    owner = f"QIR body call {index}"
    match = _QIR_CALL.fullmatch(line)
    assert match is not None
    callee = match.group(1).removeprefix("__quantum__qis__")
    operands = match.group(2).split(", ")
    if callee in _QIR_ROTATIONS:
        angle = _QIR_DOUBLE.fullmatch(operands[0])
        if angle is None:
            raise TargetConformanceError(f"{owner} has a malformed angle operand")
        theta = float(angle.group(1))
        if not math.isfinite(theta):
            raise TargetConformanceError(f"{owner} has a non-finite angle")
        wires = tuple(
            _qir_operand(item, owner=owner, writeonly=False) for item in operands[1:]
        )
        if len(wires) != 1:
            raise TargetConformanceError(f"{owner} has the wrong operand count")
        if wires[0] >= n_wires:
            raise TargetConformanceError(f"{owner} references an out-of-range qubit")
        return Instruction(_QIR_ROTATIONS[callee], wires, params={"theta": theta})
    try:
        opcode, arity = _QIR_NATIVE_GATES[callee]
    except KeyError as error:
        raise TargetConformanceError(
            f"{owner} uses unsupported QIS function {callee!r}"
        ) from error
    wires = tuple(_qir_operand(item, owner=owner, writeonly=False) for item in operands)
    if len(wires) != arity:
        raise TargetConformanceError(f"{owner} has the wrong operand count")
    if any(wire >= n_wires for wire in wires):
        raise TargetConformanceError(f"{owner} references an out-of-range qubit")
    return Instruction(opcode, wires)


def _qir_attributes(text: str) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = _QIR_ATTRIBUTE_GROUP.fullmatch(line)
        if match is None:
            continue
        if match.group(1) in groups:
            raise TargetConformanceError("qir-2.0 repeats an attribute group")
        groups[match.group(1)] = dict(_QIR_ATTRIBUTE.findall(match.group(2)))
    return groups


def _parse_qir(text: str, *, template: CircuitIR) -> CircuitIR:
    lines = text.splitlines()
    if not lines or lines[0] != _QIR_PREAMBLE:
        raise TargetConformanceError("qir-2.0 header is not canonical")

    labels: dict[str, str] = {}
    for line in lines[1:]:
        if line == "":
            continue
        if line.startswith("define "):
            break
        match = _QIR_LABEL.fullmatch(line)
        if match is None:
            raise TargetConformanceError("qir-2.0 label constant is not canonical")
        name, length, label = (
            f"@{match.group(1)}",
            int(match.group(2)),
            match.group(3),
        )
        if name in labels or length != len(label.encode("utf-8")) + 1:
            raise TargetConformanceError("qir-2.0 label constant is inconsistent")
        labels[name] = label
    if sorted(labels) != [f"@{index}" for index in range(len(labels))]:
        raise TargetConformanceError("qir-2.0 label numbering is not consecutive")
    if len(set(labels.values())) != len(labels):
        raise TargetConformanceError("qir-2.0 reuses an output label string")

    definitions = [
        index for index, line in enumerate(lines) if line.startswith("define ")
    ]
    if len(definitions) != 1:
        raise TargetConformanceError("qir-2.0 must define exactly one entry point")
    start = definitions[0]
    match = _QIR_DEFINE.fullmatch(lines[start])
    if match is None:
        raise TargetConformanceError("qir-2.0 entry point signature is not canonical")
    try:
        end = lines.index("}", start)
    except ValueError as error:
        raise TargetConformanceError("qir-2.0 entry point is not terminated") from error

    groups = _qir_attributes(text)
    try:
        entry_attributes = groups[match.group(2)]
    except KeyError as error:
        raise TargetConformanceError(
            "qir-2.0 entry point attribute group is missing"
        ) from error
    if (
        "entry_point" not in entry_attributes
        or entry_attributes.get("qir_profiles") != "base_profile"
        or "output_labeling_schema" not in entry_attributes
    ):
        raise TargetConformanceError("qir-2.0 entry point attributes are incomplete")
    if int(entry_attributes.get("required_num_qubits", "-1")) != template.n_wires:
        raise TargetConformanceError(
            "qir-2.0 required qubit count does not match the legalized circuit"
        )

    blocks: list[tuple[str, list[str]]] = []
    for line in lines[start + 1 : end]:
        if line == "":
            continue
        header = _QIR_BLOCK.fullmatch(line)
        if header is not None:
            blocks.append((header.group(1), []))
        elif not blocks:
            raise TargetConformanceError("qir-2.0 statement outside a basic block")
        else:
            blocks[-1][1].append(line.strip())
    if len(blocks) != 4 or len({name for name, _ in blocks}) != 4:
        raise TargetConformanceError("qir-2.0 entry point needs four basic blocks")
    names = [name for name, _ in blocks]
    for position in range(3):
        statements = blocks[position][1]
        branch = _QIR_BRANCH.fullmatch(statements[-1]) if statements else None
        if branch is None or branch.group(1) != names[position + 1]:
            raise TargetConformanceError(
                "qir-2.0 basic blocks are not an unconditional chain"
            )

    entry_statements = blocks[0][1][:-1]
    if entry_statements != ["call void @__quantum__rt__initialize(ptr null)"]:
        raise TargetConformanceError("qir-2.0 initialization block is not canonical")
    body_lines = blocks[1][1][:-1]
    measurement_lines = blocks[2][1][:-1]
    output_lines = blocks[3][1]
    if output_lines[-1] != "ret i64 0":
        raise TargetConformanceError("qir-2.0 exit code is not a static zero")
    output_lines = output_lines[:-1]

    instructions = tuple(
        _qir_body_instruction(line, index=index, n_wires=template.n_wires)
        for index, line in enumerate(body_lines)
    )

    result_wires = _allocated_result_wires(template)
    measured = (
        tuple(range(template.n_wires)) if result_wires is None else tuple(result_wires)
    )
    for index, line in enumerate(measurement_lines):
        qubit, result = _qir_mz_operands(line, owner=f"QIR measurement call {index}")
        if result != index:
            raise TargetConformanceError(
                "qir-2.0 measurement results are not consecutively indexed"
            )
        if qubit != measured[index]:
            raise TargetConformanceError(
                "qir-2.0 measurement target does not match the legalized projection"
            )
    if len(measurement_lines) != len(measured):
        raise TargetConformanceError(
            "qir-2.0 measurement count does not match the legalized projection"
        )
    if int(entry_attributes.get("required_num_results", "-1")) != len(measured):
        raise TargetConformanceError(
            "qir-2.0 required result count does not match the legalized projection"
        )

    if len(output_lines) != len(measured) + 1:
        raise TargetConformanceError("qir-2.0 output block is not canonical")
    kind, operands, label_tokens = _qir_record_operands(output_lines[0])
    if kind != "tuple_record_output" or operands != (len(measured),):
        raise TargetConformanceError("qir-2.0 tuple record is not canonical")
    if (
        _qir_label_operand(label_tokens[0], labels=labels, owner="qir-2.0 tuple record")
        != "t0"
    ):
        raise TargetConformanceError("qir-2.0 tuple label is not canonical")
    used_labels = {"t0"}
    for position, line in enumerate(output_lines[1:]):
        kind, operands, label_tokens = _qir_record_operands(line)
        if kind != "result_record_output" or operands != (position,):
            raise TargetConformanceError("qir-2.0 result record is not canonical")
        label = _qir_label_operand(
            label_tokens[0], labels=labels, owner="qir-2.0 result record"
        )
        if label != f"r{position}":
            raise TargetConformanceError("qir-2.0 result label is not canonical")
        used_labels.add(label)
    if used_labels != set(labels.values()):
        raise TargetConformanceError("qir-2.0 leaves an output label unreferenced")

    declared: set[str] = set()
    for line in text.splitlines():
        declaration = _QIR_DECLARE.fullmatch(line)
        if declaration is None:
            continue
        name, signature, marker = declaration.groups()
        if name in declared:
            raise TargetConformanceError("qir-2.0 repeats a function declaration")
        try:
            expected_signature = _QIR_SIGNATURES[name]
        except KeyError as error:
            raise TargetConformanceError(
                f"qir-2.0 declares an unknown function {name!r}"
            ) from error
        if signature != expected_signature:
            raise TargetConformanceError(
                f"qir-2.0 declaration of {name!r} has the wrong signature"
            )
        irreversible = name == _QIR_IRREVERSIBLE_CALL
        if (marker is not None) != irreversible:
            raise TargetConformanceError(
                f"qir-2.0 declaration of {name!r} has the wrong irreversibility marker"
            )
        if marker is not None and groups.get(marker) != {"irreversible": ""}:
            raise TargetConformanceError(
                "qir-2.0 irreversible marker does not name a canonical attribute group"
            )
        declared.add(name)
    used: set[str] = set()
    for line in (
        *entry_statements,
        *body_lines,
        *measurement_lines,
        *output_lines,
    ):
        call = _QIR_CALL.fullmatch(line)
        if call is not None:
            used.add(call.group(1))
    if declared != used:
        raise TargetConformanceError(
            "qir-2.0 declarations do not match the calls in the entry point"
        )

    flags: dict[str, tuple[str, str]] = {}
    referenced: list[str] = []
    for line in text.splitlines():
        if line.startswith("!llvm.module.flags"):
            referenced = re.findall(r"!([0-9]+)", line.split("=", 1)[1])
            continue
        match = _QIR_MODULE_FLAG.fullmatch(line)
        if match is None:
            continue
        value = match.group(5) if match.group(5) is not None else match.group(6)
        flags[match.group(3)] = (match.group(2), value)
    if flags != _QIR_REQUIRED_FLAGS:
        raise TargetConformanceError("qir-2.0 module flags are not canonical")
    if referenced != [f"{index}" for index in range(len(referenced))]:
        raise TargetConformanceError("qir-2.0 module flag numbering is not consecutive")
    if len(referenced) != len(flags):
        raise TargetConformanceError("qir-2.0 module flag list is inconsistent")
    return _reconstructed_program(template, instructions)


def _conformance_identity(
    emission: TargetEmissionResult,
    reconstructed: CircuitIR,
) -> str:
    payload = {
        "emission_identity": emission.emission_identity,
        "profile": emission.profile,
        "parser": (
            "flagquantum.compiler.target_conformance.v2"
            if _allocated_result_wires(reconstructed) is not None
            else "flagquantum.compiler.target_conformance.v1"
        ),
        "reconstructed_circuit_hash": reconstructed.content_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_target_emission(
    emission: TargetEmissionResult,
    legalization: TargetLegalizationResult,
) -> TargetConformanceResult:
    """Verify identities and strictly parse one canonical emitted payload."""

    if not isinstance(emission, TargetEmissionResult):
        raise TypeError("target conformance requires a TargetEmissionResult")
    if not isinstance(legalization, TargetLegalizationResult):
        raise TypeError("target conformance requires a TargetLegalizationResult")
    try:
        expected = emit_legalized_target(legalization, profile=emission.profile)
    except CompilationError as error:
        raise TargetConformanceError(
            f"cannot reproduce target emission: {error}"
        ) from error
    if emission != expected:
        raise TargetConformanceError(
            "target emission payload or identity evidence does not match legalization"
        )

    if emission.profile in {"openqasm-2.0", "openqasm-3.0"}:
        reconstructed = _parse_openqasm(
            emission.text,
            profile=emission.profile,
            template=legalization.program,
        )
    elif emission.profile == "qcis-1.0":
        reconstructed = _parse_qcis(emission.text, template=legalization.program)
    elif emission.profile == "qir-2.0":
        reconstructed = _parse_qir(emission.text, template=legalization.program)
    else:
        raise TargetConformanceError(
            f"no conformance parser for profile {emission.profile!r}"
        )
    return TargetConformanceResult(
        emission_identity=emission.emission_identity,
        reconstructed_program=reconstructed,
        reconstructed_circuit_hash=reconstructed.content_hash,
        parsed_operation_count=len(reconstructed.instructions),
        conformance_identity=_conformance_identity(emission, reconstructed),
    )


__all__ = (
    "TargetConformanceError",
    "TargetConformanceResult",
    "verify_target_emission",
)
