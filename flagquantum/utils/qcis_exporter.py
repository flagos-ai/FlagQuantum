"""QCIS exporter for FlagQuantum circuits and IR.

QCIS-native cloud backends consume a small native instruction set. This module
serializes FlagQuantum's unified IR directly to that native form without a QASM
round trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..core.ir import CircuitIR
from ..core.parameters import is_parameterized_value, parameter_names_in_value

_PI = round(math.pi, 6)


@dataclass(frozen=True)
class QCISInstruction:
    """A single QCIS-native instruction."""

    name: str
    wires: tuple[int, ...]
    args: tuple[float | int, ...] = ()

    def __str__(self) -> str:
        parts = [self.name.upper()]
        parts.extend(f"Q{wire}" for wire in self.wires)
        for value in self.args:
            if (
                isinstance(value, float)
                and self.name.lower() == "rz"
                and abs(abs(value) - math.pi) < 1e-12
            ):
                value = math.copysign(math.pi - 1e-10, value)
            parts.append(str(value))
        return " ".join(parts)


def _as_ir(program: Any) -> CircuitIR:
    if isinstance(program, CircuitIR):
        return program
    if hasattr(program, "to_ir"):
        return program.to_ir()
    raise TypeError("QCIS export expects a FlagQuantum Circuit or CircuitIR.")


def _to_float(value: Any) -> float:
    if is_parameterized_value(value):
        names = ", ".join(parameter_names_in_value(value))
        raise ValueError(
            f"Unbound circuit parameter(s): {names}. Call bind_parameters first."
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("QCIS export requires scalar gate parameters.")
        return float(value.detach().reshape(()).cpu())
    return float(value)


def _param(params: Mapping[str, Any], *names: str) -> tuple[float, ...]:
    values = []
    for name in names:
        if name not in params:
            raise ValueError(f"Missing parameter {name!r} for QCIS export.")
        values.append(_to_float(params[name]))
    return tuple(values)


def _one(name: str, wire: int, *args: float | int) -> list[QCISInstruction]:
    return [QCISInstruction(name, (wire,), tuple(args))]


def _cx(control: int, target: int) -> list[QCISInstruction]:
    return [
        QCISInstruction("y2m", (target,)),
        QCISInstruction("cz", (control, target)),
        QCISInstruction("y2p", (target,)),
    ]


def _decompose(
    name: str, wires: tuple[int, ...], params: Mapping[str, Any]
) -> list[QCISInstruction]:
    if name in {"measure", "measure_allz", "barrier"}:
        return []
    if name in {"i", "id"}:
        return _one("i", wires[0], 60)
    if name == "x":
        return _one("x2p", wires[0]) + _one("x2p", wires[0])
    if name == "y":
        return _one("y2p", wires[0]) + _one("y2p", wires[0])
    if name == "z":
        return _one("rz", wires[0], _PI)
    if name == "h":
        return _one("y2m", wires[0]) + _one("rz", wires[0], _PI)
    if name == "sx":
        return _one("x2p", wires[0])
    if name == "sxdg":
        return _one("x2m", wires[0])
    if name == "s":
        return _one("rz", wires[0], _PI / 2)
    if name == "sdg":
        return _one("rz", wires[0], -_PI / 2)
    if name == "t":
        return _one("rz", wires[0], _PI / 4)
    if name == "tdg":
        return _one("rz", wires[0], -_PI / 4)
    if name == "rx":
        (theta,) = _param(params, "theta")
        return (
            _one("y2m", wires[0]) + _one("rz", wires[0], theta) + _one("y2p", wires[0])
        )
    if name == "ry":
        (theta,) = _param(params, "theta")
        return (
            _one("x2p", wires[0]) + _one("rz", wires[0], theta) + _one("x2m", wires[0])
        )
    if name == "rz":
        (theta,) = _param(params, "theta")
        return _one("rz", wires[0], theta)
    if name == "phase":
        (theta,) = _param(params, "theta")
        return _one("rz", wires[0], theta)
    if name in {"u", "u3"}:
        theta, phi, lbd = _param(params, "theta", "phi", "lbd")
        return (
            _one("rz", wires[0], lbd)
            + _one("x2p", wires[0])
            + _one("rz", wires[0], theta)
            + _one("x2m", wires[0])
            + _one("rz", wires[0], phi)
        )
    if name == "u1":
        (theta,) = _param(params, "theta")
        return _one("rz", wires[0], theta)
    if name == "u2":
        phi, lbd = _param(params, "phi", "lbd")
        return _decompose("u3", wires, {"theta": math.pi / 2, "phi": phi, "lbd": lbd})
    if name == "cx":
        return _cx(wires[0], wires[1])
    if name == "cy":
        return (
            _one("x2p", wires[1])
            + [QCISInstruction("cz", wires)]
            + _one("x2m", wires[1])
        )
    if name == "cz":
        return [QCISInstruction("cz", wires)]
    if name == "swap":
        return (
            _cx(wires[0], wires[1]) + _cx(wires[1], wires[0]) + _cx(wires[0], wires[1])
        )
    if name == "rzz":
        (theta,) = _param(params, "theta")
        return (
            _cx(wires[0], wires[1])
            + _one("rz", wires[1], theta)
            + _cx(wires[0], wires[1])
        )
    if name == "rxx":
        (theta,) = _param(params, "theta")
        return (
            _decompose("h", (wires[0],), {})
            + _decompose("h", (wires[1],), {})
            + _decompose("rzz", wires, {"theta": theta})
            + _decompose("h", (wires[0],), {})
            + _decompose("h", (wires[1],), {})
        )
    if name == "ryy":
        (theta,) = _param(params, "theta")
        return (
            _decompose("rx", (wires[0],), {"theta": math.pi / 2})
            + _decompose("rx", (wires[1],), {"theta": math.pi / 2})
            + _decompose("rzz", wires, {"theta": theta})
            + _decompose("rx", (wires[0],), {"theta": -math.pi / 2})
            + _decompose("rx", (wires[1],), {"theta": -math.pi / 2})
        )
    if name == "ccx":
        q0, q1, q2 = wires
        out: list[QCISInstruction] = []
        for gate, gate_wires, gate_params in (
            ("h", (q2,), {}),
            ("cx", (q1, q2), {}),
            ("tdg", (q2,), {}),
            ("cx", (q0, q2), {}),
            ("t", (q2,), {}),
            ("cx", (q1, q2), {}),
            ("tdg", (q2,), {}),
            ("cx", (q0, q2), {}),
            ("t", (q1,), {}),
            ("t", (q2,), {}),
            ("h", (q2,), {}),
            ("cx", (q0, q1), {}),
            ("t", (q0,), {}),
            ("tdg", (q1,), {}),
            ("cx", (q0, q1), {}),
        ):
            out.extend(_decompose(gate, gate_wires, gate_params))
        return out
    raise NotImplementedError(f"Gate {name!r} has no QCIS decomposition.")


def export_to_qcis_str(program: Any) -> str:
    """Export a FlagQuantum circuit or IR to a QCIS instruction string."""

    ir = _as_ir(program)
    from ..compiler.operator_lowering import validate_lowering

    validate_lowering(ir, "qcis")
    lines: list[str] = []
    for instruction in ir.instructions:
        if instruction.matrix is not None:
            raise NotImplementedError(
                "QCIS export does not support arbitrary matrix gates."
            )
        for native in _decompose(
            instruction.name.lower(), instruction.wires, instruction.params
        ):
            lines.append(str(native))
    return "\n".join(lines)


__all__ = ["QCISInstruction", "export_to_qcis_str"]
