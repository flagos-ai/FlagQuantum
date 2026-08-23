"""Executable semantic certification for the optional Qiskit boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

from ...circuit import Circuit
from ...core.ir import ensure_circuit_ir
from .conversion import from_qiskit, to_qiskit
from .models import QiskitDependencyError


@dataclass(frozen=True)
class QiskitConformanceCaseResult:
    """One statevector and round-trip semantic comparison."""

    name: str
    passed: bool
    maximum_absolute_error: float
    source_fingerprint: str
    round_trip_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible case record."""

        return {
            "name": self.name,
            "passed": self.passed,
            "maximum_absolute_error": self.maximum_absolute_error,
            "source_fingerprint": self.source_fingerprint,
            "round_trip_fingerprint": self.round_trip_fingerprint,
        }


@dataclass(frozen=True)
class QiskitConformanceResult:
    """Aggregate Qiskit interoperability certification result."""

    schema: str
    qiskit_version: str
    cases: tuple[QiskitConformanceCaseResult, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned machine-readable certification payload."""

        return {
            "schema": self.schema,
            "qiskit_version": self.qiskit_version,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
        }


def semantic_fingerprint(program: Any) -> str:
    """Hash executable circuit semantics while excluding transport provenance."""

    ir = ensure_circuit_ir(program)
    interop = ir.metadata.get("interop", {})
    global_phase = (
        interop.get("global_phase", 0.0) if isinstance(interop, dict) else 0.0
    )
    payload = {
        "schema": "flagquantum_qiskit_semantics_v1",
        "ir_version": ir.version,
        "n_wires": ir.n_wires,
        "instructions": ir.to_dict()["instructions"],
        "global_phase": 0.0 if global_phase is None else global_phase,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def qiskit_statevector_to_flagquantum(statevector: Any, n_wires: int) -> torch.Tensor:
    """Convert Qiskit's little-endian amplitude order to FlagQuantum wire order."""

    n_wires = int(n_wires)
    if n_wires <= 0:
        raise ValueError("n_wires must be positive")
    tensor = torch.as_tensor(statevector)
    expected = 2**n_wires
    if tensor.numel() != expected:
        raise ValueError(
            f"statevector has {tensor.numel()} amplitudes, expected {expected}"
        )
    axes = tuple(reversed(range(n_wires)))
    return tensor.reshape((2,) * n_wires).permute(axes).reshape(-1)


def _cases() -> tuple[tuple[str, Circuit], ...]:
    return (
        (
            "asymmetric_wire_order",
            Circuit(3, dtype=torch.complex128)
            .x(0)
            .ry(2, theta=0.371)
            .cx(0, 1)
            .rz(1, theta=-0.219),
        ),
        (
            "controlled_rotations",
            Circuit(3, dtype=torch.complex128)
            .h(1)
            .crx(1, 2, theta=0.417)
            .cphase(2, 0, theta=-0.193)
            .swap(0, 1),
        ),
        (
            "three_qubit_controls",
            Circuit(3, dtype=torch.complex128).x(0).x(1).ccx(0, 1, 2).cswap(2, 0, 1),
        ),
    )


def run_qiskit_conformance(*, atol: float = 1e-10) -> QiskitConformanceResult:
    """Execute deterministic statevector and IR round-trip conformance cases."""

    try:
        qiskit = import_module("qiskit")
        statevector_type = import_module("qiskit.quantum_info").Statevector
    except ImportError as exc:
        raise QiskitDependencyError(
            "Qiskit conformance requires `pip install 'flagquantum[qiskit]'`."
        ) from exc

    results: list[QiskitConformanceCaseResult] = []
    for name, source in _cases():
        source_ir = source.to_ir()
        qiskit_circuit = to_qiskit(source_ir)
        round_trip_ir = from_qiskit(qiskit_circuit)
        flagquantum_state = Circuit.from_ir(
            round_trip_ir, dtype=torch.complex128
        ).state()[0]
        qiskit_state = qiskit_statevector_to_flagquantum(
            statevector_type.from_instruction(qiskit_circuit).data,
            source_ir.n_wires,
        ).to(dtype=torch.complex128)
        maximum_error = float(
            torch.max(torch.abs(flagquantum_state - qiskit_state)).item()
        )
        source_fingerprint = semantic_fingerprint(source_ir)
        round_trip_fingerprint = semantic_fingerprint(round_trip_ir)
        results.append(
            QiskitConformanceCaseResult(
                name=name,
                passed=(
                    maximum_error <= float(atol)
                    and source_fingerprint == round_trip_fingerprint
                ),
                maximum_absolute_error=maximum_error,
                source_fingerprint=source_fingerprint,
                round_trip_fingerprint=round_trip_fingerprint,
            )
        )
    cases = tuple(results)
    return QiskitConformanceResult(
        schema="flagquantum_qiskit_conformance_v1",
        qiskit_version=str(qiskit.__version__),
        cases=cases,
        passed=all(case.passed for case in cases),
    )


__all__ = (
    "QiskitConformanceCaseResult",
    "QiskitConformanceResult",
    "qiskit_statevector_to_flagquantum",
    "run_qiskit_conformance",
    "semantic_fingerprint",
)
