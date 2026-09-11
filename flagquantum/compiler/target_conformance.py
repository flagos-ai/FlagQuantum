"""Strict conformance checks for Compiler-emitted target text."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace

from ..core.ir import CircuitIR, Instruction, MeasurementNode
from ..errors import CompilationError
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

_FIXED_QASM_GATES = {
    "id": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "sx": "sx",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "ccx": "ccx",
    "cswap": "cswap",
}
_PARAMETERIZED_QASM_GATES = {
    "rx": ("rx", ("theta",)),
    "ry": ("ry", ("theta",)),
    "rz": ("rz", ("theta",)),
    "u1": ("u1", ("theta",)),
    "u2": ("u2", ("phi", "lbd")),
    "u3": ("u3", ("theta", "phi", "lbd")),
    "p": ("phase", ("theta",)),
    "cu1": ("cphase", ("theta",)),
    "cp": ("cphase", ("theta",)),
    "crx": ("crx", ("theta",)),
    "cry": ("cry", ("theta",)),
    "crz": ("crz", ("theta",)),
}


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
    return Instruction(opcode, wires, params=dict(zip(parameter_names, values)))


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
