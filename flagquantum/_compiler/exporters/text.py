"""Private deterministic emitters for the Phase 2 static target profile."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum

from ..bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from ..diagnostics import Diagnostic, DiagnosticCode
from ..ir.modules import QuantumModule
from ..ir.operations import FrozenAttributes
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.verifier import verify_module


class TextEmissionStatus(str, Enum):
    EMITTED_EXACT = "emitted_exact"
    UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
    INVALID_MODULE = "invalid_module"


@dataclass(frozen=True)
class TextEmissionResult:
    status: TextEmissionStatus
    profile: str
    source_program_identity: str | None = None
    text: str | None = None
    content_hash: str | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is TextEmissionStatus.EMITTED_EXACT


@dataclass(frozen=True)
class _StaticInstruction:
    name: str
    wires: tuple[int, ...]
    theta: str | None = None


class _EmissionError(Exception):
    pass


def _failure(
    profile: str,
    status: TextEmissionStatus,
    message: str,
    *,
    source_program_identity: str | None = None,
) -> TextEmissionResult:
    return TextEmissionResult(
        status,
        profile,
        source_program_identity=source_program_identity,
        diagnostics=(
            Diagnostic(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                message,
                notes=("restricted text emission never performs lossy recovery",),
            ),
        ),
    )


def _static_scalar(value: object) -> float | int:
    if isinstance(value, (RuntimeBindingRef, SymbolicParameter, SymbolicExpression)):
        raise _EmissionError(
            "unbound parameters are outside the restricted-static profile"
        )
    if isinstance(value, bool) or isinstance(value, complex):
        raise _EmissionError("gate parameters must be real numeric scalars")
    if isinstance(value, (int, float)):
        number = value
    elif isinstance(value, FrozenAttributes) and value.get("kind") == "tensor":
        if tuple(value.get("shape", ())) not in {(), (1,)}:
            raise _EmissionError("gate tensor parameters must contain one scalar")
        data = value.get("data")
        if isinstance(data, tuple):
            if len(data) != 1:
                raise _EmissionError("gate tensor parameters must contain one scalar")
            data = data[0]
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise _EmissionError("gate tensor parameters must be real numeric scalars")
        number = data
    else:
        raise _EmissionError("gate parameters must be static real numeric scalars")
    if not math.isfinite(float(number)):
        raise _EmissionError("gate parameters must be finite")
    return number


def _format_number(value: object) -> str:
    number = _static_scalar(value)
    if float(number) == 0.0:
        return "0"
    return format(float(number), ".17g")


def _extract(
    module: object, profile: str
) -> tuple[int, tuple[_StaticInstruction, ...]] | TextEmissionResult:
    if not isinstance(module, QuantumModule):
        return _failure(
            profile, TextEmissionStatus.INVALID_MODULE, "emitter requires QuantumModule"
        )
    verification = verify_module(module, circuit_ir_v1_schema_registry())
    if not verification.ok:
        return TextEmissionResult(
            TextEmissionStatus.INVALID_MODULE,
            profile,
            source_program_identity=module.program_identity,
            diagnostics=verification.diagnostics,
        )
    if len(module.body.blocks) != 1:
        return _failure(
            profile,
            TextEmissionStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            "one entry block is required",
            source_program_identity=module.program_identity,
        )
    block = module.body.blocks[0]
    value_wires = {argument.id: wire for wire, argument in enumerate(block.arguments)}
    instructions: list[_StaticInstruction] = []
    try:
        for operation in block.operations:
            if operation.name not in {
                "quantum.rx",
                "quantum.ry",
                "quantum.rz",
                "quantum.cx",
            }:
                raise _EmissionError(
                    f"operation {operation.name!r} is outside the target profile"
                )
            wires = tuple(value_wires[operand.id] for operand in operation.operands)
            theta = None
            if operation.name != "quantum.cx":
                theta = _format_number(operation.attributes["theta"])
            instructions.append(
                _StaticInstruction(
                    operation.name.removeprefix("quantum."), wires, theta
                )
            )
            for result, wire in zip(operation.results, wires):
                value_wires[result.id] = wire
    except (KeyError, _EmissionError) as exc:
        return _failure(
            profile,
            TextEmissionStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            str(exc),
            source_program_identity=module.program_identity,
        )
    return len(block.arguments), tuple(instructions)


def _success(
    module: QuantumModule, profile: str, lines: list[str]
) -> TextEmissionResult:
    text = "\n".join(lines) + "\n"
    return TextEmissionResult(
        TextEmissionStatus.EMITTED_EXACT,
        profile,
        source_program_identity=module.program_identity,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def emit_openqasm2(module: object) -> TextEmissionResult:
    profile = "openqasm_2_static_rx_ry_rz_cx_v1"
    extracted = _extract(module, profile)
    if isinstance(extracted, TextEmissionResult):
        return extracted
    n_qubits, instructions = extracted
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];"]
    for item in instructions:
        operands = ",".join(f"q[{wire}]" for wire in item.wires)
        gate = item.name if item.theta is None else f"{item.name}({item.theta})"
        lines.append(f"{gate} {operands};")
    return _success(module, profile, lines)  # type: ignore[arg-type]


def emit_openqasm3(module: object) -> TextEmissionResult:
    profile = "openqasm_3_static_rx_ry_rz_cx_v1"
    extracted = _extract(module, profile)
    if isinstance(extracted, TextEmissionResult):
        return extracted
    n_qubits, instructions = extracted
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n_qubits}] q;"]
    for item in instructions:
        operands = ", ".join(f"q[{wire}]" for wire in item.wires)
        gate = item.name if item.theta is None else f"{item.name}({item.theta})"
        lines.append(f"{gate} {operands};")
    return _success(module, profile, lines)  # type: ignore[arg-type]


def emit_qcis_v1(module: object) -> TextEmissionResult:
    profile = "flagquantum_qcis_rx_ry_rz_cx_v1"
    extracted = _extract(module, profile)
    if isinstance(extracted, TextEmissionResult):
        return extracted
    _, instructions = extracted
    lines: list[str] = []
    for item in instructions:
        if item.name == "rx":
            lines.extend(
                (
                    f"Y2M Q{item.wires[0]}",
                    f"RZ Q{item.wires[0]} {item.theta}",
                    f"Y2P Q{item.wires[0]}",
                )
            )
        elif item.name == "ry":
            lines.extend(
                (
                    f"X2P Q{item.wires[0]}",
                    f"RZ Q{item.wires[0]} {item.theta}",
                    f"X2M Q{item.wires[0]}",
                )
            )
        elif item.name == "rz":
            lines.append(f"RZ Q{item.wires[0]} {item.theta}")
        else:
            control, target = item.wires
            lines.extend(
                (f"Y2M Q{target}", f"CZ Q{control} Q{target}", f"Y2P Q{target}")
            )
    return _success(module, profile, lines)  # type: ignore[arg-type]


__all__ = [
    "TextEmissionResult",
    "TextEmissionStatus",
    "emit_openqasm2",
    "emit_openqasm3",
    "emit_qcis_v1",
]
