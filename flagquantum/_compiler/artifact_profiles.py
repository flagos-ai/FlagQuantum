"""Closed executable-artifact profiles and payload validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from .ir.operations import FrozenAttributes
from .target_capabilities import ArtifactFormat, ArtifactProfile
from .target_ir import TargetIR, TargetOperation


class ArtifactEncoding(str, Enum):
    UTF8_TEXT = "utf8_text"


@dataclass(frozen=True)
class ArtifactProfileDefinition:
    profile: ArtifactProfile
    media_type: str
    encoding: ArtifactEncoding
    portable: bool = True


_DEFINITIONS = (
    ArtifactProfileDefinition(
        ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),
        "application/vnd.flagquantum.target-ir+json",
        ArtifactEncoding.UTF8_TEXT,
    ),
    ArtifactProfileDefinition(
        ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),
        "text/x-openqasm;version=2.0",
        ArtifactEncoding.UTF8_TEXT,
    ),
    ArtifactProfileDefinition(
        ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0"),
        "text/x-openqasm;version=3.0",
        ArtifactEncoding.UTF8_TEXT,
    ),
    ArtifactProfileDefinition(
        ArtifactProfile(ArtifactFormat.QCIS_1, "1.0"),
        "text/x-qcis;version=1.0",
        ArtifactEncoding.UTF8_TEXT,
    ),
)

ARTIFACT_PROFILE_REGISTRY: Mapping[ArtifactProfile, ArtifactProfileDefinition] = (
    MappingProxyType({item.profile: item for item in _DEFINITIONS})
)


def artifact_profile_definition(
    profile: ArtifactProfile,
) -> ArtifactProfileDefinition:
    if not isinstance(profile, ArtifactProfile):
        raise ValueError("artifact profile must use ArtifactProfile")
    try:
        return ARTIFACT_PROFILE_REGISTRY[profile]
    except KeyError as exc:
        raise ValueError("artifact profile is not registered") from exc


def _number(value: object) -> str:
    if isinstance(value, (RuntimeBindingRef, SymbolicParameter, SymbolicExpression)):
        raise ValueError("text artifact requires statically bound parameters")
    if isinstance(value, bool) or isinstance(value, complex):
        raise ValueError("text artifact parameter must be a real scalar")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, FrozenAttributes) and value.get("kind") == "tensor":
        shape = tuple(value.get("shape", ()))
        data = value.get("data")
        if shape not in {(), (1,)}:
            raise ValueError("text artifact tensor parameter must contain one scalar")
        if isinstance(data, tuple):
            if len(data) != 1:
                raise ValueError(
                    "text artifact tensor parameter must contain one scalar"
                )
            data = data[0]
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ValueError("text artifact tensor parameter must be a real scalar")
        number = float(data)
    else:
        raise ValueError("text artifact parameter must be a static real scalar")
    if not math.isfinite(number):
        raise ValueError("text artifact parameter must be finite")
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _qasm_payload(target_ir: TargetIR, *, version: int) -> bytes:
    if target_ir.required_results or target_ir.requested_shots is not None:
        raise ValueError(
            "static QASM artifact cannot represent result or shot requests"
        )
    used_qubits = (
        *target_ir.logical_to_physical,
        *(
            item
            for operation in target_ir.operations
            for item in operation.physical_qubits
        ),
    )
    qubit_count = max(used_qubits, default=-1) + 1
    if version == 2:
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{qubit_count}];",
        ]
        separator = ","
    else:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{qubit_count}] q;",
        ]
        separator = ", "
    for operation in target_ir.operations:
        if operation.operation not in {"rx", "ry", "rz", "cx"}:
            raise ValueError("operation is outside the static QASM artifact profile")
        operands = separator.join(f"q[{item}]" for item in operation.physical_qubits)
        gate = operation.operation
        if gate != "cx":
            gate = f"{gate}({_number(operation.attributes['theta'])})"
        lines.append(f"{gate} {operands};")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _qcis_lines(operation: TargetOperation) -> tuple[str, ...]:
    if operation.operation == "rx":
        wire = operation.physical_qubits[0]
        return (
            f"Y2M Q{wire}",
            f"RZ Q{wire} {_number(operation.attributes['theta'])}",
            f"Y2P Q{wire}",
        )
    if operation.operation == "ry":
        wire = operation.physical_qubits[0]
        return (
            f"X2P Q{wire}",
            f"RZ Q{wire} {_number(operation.attributes['theta'])}",
            f"X2M Q{wire}",
        )
    if operation.operation == "rz":
        wire = operation.physical_qubits[0]
        return (f"RZ Q{wire} {_number(operation.attributes['theta'])}",)
    if operation.operation == "cx":
        control, target = operation.physical_qubits
        return (f"Y2M Q{target}", f"CZ Q{control} Q{target}", f"Y2P Q{target}")
    raise ValueError("operation is outside the QCIS artifact profile")


def _qcis_payload(target_ir: TargetIR) -> bytes:
    if target_ir.required_results or target_ir.requested_shots is not None:
        raise ValueError(
            "static QCIS artifact cannot represent result or shot requests"
        )
    lines = [
        line for operation in target_ir.operations for line in _qcis_lines(operation)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _runtime_plan_payload(target_ir: TargetIR) -> bytes:
    return json.dumps(
        target_ir.canonical(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def validate_artifact_payload(
    profile: ArtifactProfile,
    payload: bytes,
    target_ir: TargetIR,
) -> None:
    """Require the payload to be the exact canonical encoding of TargetIR."""

    artifact_profile_definition(profile)
    if not isinstance(payload, bytes):
        raise ValueError("artifact payload must be immutable bytes")
    if not isinstance(target_ir, TargetIR):
        raise ValueError("artifact payload validation requires TargetIR")
    if profile.format is ArtifactFormat.RUNTIME_PLAN:
        expected = _runtime_plan_payload(target_ir)
    elif profile.format is ArtifactFormat.OPENQASM_2:
        expected = _qasm_payload(target_ir, version=2)
    elif profile.format is ArtifactFormat.OPENQASM_3_STATIC:
        expected = _qasm_payload(target_ir, version=3)
    elif profile.format is ArtifactFormat.QCIS_1:
        expected = _qcis_payload(target_ir)
    else:  # pragma: no cover - closed enum and registry make this defensive.
        raise ValueError("artifact format has no validator")
    if payload != expected:
        raise ValueError(
            "artifact payload does not canonically match TargetIR and profile"
        )


__all__ = [
    "ARTIFACT_PROFILE_REGISTRY",
    "ArtifactEncoding",
    "ArtifactProfileDefinition",
    "artifact_profile_definition",
    "validate_artifact_payload",
]
