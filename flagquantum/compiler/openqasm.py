"""Compile canonical FlagQuantum IR to OpenQASM 2 or OpenQASM 3 text."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.parameters import is_parameterized_value, parameter_names_in_value
from .operator_lowering import validate_lowering

_DIRECT_GATES = {
    "i": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "u1": "u1",
    "u2": "u2",
    "u3": "u3",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
    "ccx": "ccx",
    "cswap": "cswap",
}


def _format_number(value: Any) -> str:
    if is_parameterized_value(value):
        names = ", ".join(parameter_names_in_value(value))
        raise ValueError(
            f"Unbound circuit parameter(s): {names}. Call bind_parameters first."
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("OpenQASM emission requires scalar gate parameters.")
        value = value.detach().reshape(()).cpu().item()
    if not isinstance(value, Real):
        raise TypeError(f"OpenQASM gate parameter must be real, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("OpenQASM gate parameters must be finite.")
    return repr(number)


def _parameter_text(instruction: Instruction) -> str:
    schema = get_operator_schema(instruction.name)
    if schema is None:
        raise ValueError(f"OpenQASM cannot emit unknown gate {instruction.name!r}.")
    if not schema.parameters:
        return ""
    values = []
    for name in schema.parameters:
        if name not in instruction.params:
            raise ValueError(
                f"OpenQASM gate {schema.opcode!r} is missing parameter {name!r}."
            )
        values.append(_format_number(instruction.params[name]))
    return f"({', '.join(values)})"


def _gate(name: str, wires: tuple[int, ...], parameter_text: str = "") -> str:
    operands = ", ".join(f"q[{wire}]" for wire in wires)
    return f"{name}{parameter_text} {operands};"


def _rotation(name: str, theta: str, wire: int) -> str:
    return _gate(name, (wire,), f"({theta})")


def _rzz(theta: str, left: int, right: int) -> list[str]:
    return [
        _gate("cx", (left, right)),
        _rotation("rz", theta, right),
        _gate("cx", (left, right)),
    ]


def _interaction_lines(opcode: str, theta: str, wires: tuple[int, ...]) -> list[str]:
    left, right = wires
    if opcode == "rzz":
        return _rzz(theta, left, right)
    if opcode == "rxx":
        return [
            _gate("h", (left,)),
            _gate("h", (right,)),
            *_rzz(theta, left, right),
            _gate("h", (left,)),
            _gate("h", (right,)),
        ]
    half_pi = repr(math.pi / 2)
    return [
        _rotation("rx", half_pi, left),
        _rotation("rx", half_pi, right),
        *_rzz(theta, left, right),
        _rotation("rx", f"-{half_pi}", left),
        _rotation("rx", f"-{half_pi}", right),
    ]


def _instruction_lines(instruction: Instruction, *, version: float) -> list[str]:
    if instruction.matrix is not None:
        raise ValueError("OpenQASM emission does not support arbitrary matrix gates.")
    opcode = canonical_opcode(instruction.name)
    parameters = _parameter_text(instruction)
    if opcode in {"rxx", "ryy", "rzz"}:
        return _interaction_lines(opcode, parameters[1:-1], instruction.wires)
    if version == 3.0 and opcode == "u1":
        return [_gate("p", instruction.wires, parameters)]
    if version == 3.0 and opcode == "u2":
        phi, lbd = (_format_number(instruction.params[name]) for name in ("phi", "lbd"))
        return [_gate("U", instruction.wires, f"({repr(math.pi / 2)}, {phi}, {lbd})")]
    if version == 3.0 and opcode == "u3":
        return [_gate("U", instruction.wires, parameters)]
    if opcode == "phase":
        return [_gate("p" if version == 3.0 else "u1", instruction.wires, parameters)]
    if opcode == "cphase":
        return [_gate("cp" if version == 3.0 else "cu1", instruction.wires, parameters)]
    if opcode == "sx":
        if version == 3.0:
            return [_gate("sx", instruction.wires)]
        wire = instruction.wires
        return [_gate("h", wire), _gate("s", wire), _gate("h", wire)]
    if opcode == "sxdg":
        if version == 3.0:
            return [f"pow(-1) @ {_gate('sx', instruction.wires)}"]
        wire = instruction.wires
        return [_gate("h", wire), _gate("sdg", wire), _gate("h", wire)]
    try:
        target_name = _DIRECT_GATES[opcode]
    except KeyError as exc:
        raise ValueError(f"OpenQASM cannot emit gate {opcode!r}.") from exc
    return [_gate(target_name, instruction.wires, parameters)]


def _validated_ir(program: Any) -> CircuitIR:
    ir = ensure_circuit_ir(program)
    validate_lowering(ir, "qasm")
    return ir


def emit_openqasm(program: Any, *, version: float = 3.0) -> str:
    """Return deterministic OpenQASM text for a canonical FlagQuantum program."""

    if version not in {2.0, 3.0}:
        raise ValueError("OpenQASM version must be 2.0 or 3.0.")
    ir = _validated_ir(program)
    if version == 2.0:
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{ir.n_wires}];",
            f"creg c[{ir.n_wires}];",
        ]
    else:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{ir.n_wires}] q;",
            f"bit[{ir.n_wires}] c;",
        ]
    for instruction in ir.instructions:
        lines.extend(_instruction_lines(instruction, version=version))
    if version == 2.0:
        lines.extend(f"measure q[{wire}] -> c[{wire}];" for wire in range(ir.n_wires))
    else:
        lines.append("c = measure q;")
    return "\n".join(lines)


__all__ = ("emit_openqasm",)
