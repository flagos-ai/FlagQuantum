"""Executable semantic certification for the optional Qiskit boundary."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

from ...circuit import Circuit
from ...core.ir import ensure_circuit_ir
from ..conformance import (
    InteropConformanceResult,
    InteropRejectionCase,
    InteropRoundTripCase,
    run_adapter_conformance,
)
from ..conformance import (
    semantic_fingerprint as framework_neutral_fingerprint,
)
from .adapter import QISKIT_ADAPTER
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
    adapter_contract: InteropConformanceResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned machine-readable certification payload."""

        payload = {
            "schema": self.schema,
            "qiskit_version": self.qiskit_version,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
        }
        if self.adapter_contract is not None:
            payload["adapter_contract"] = self.adapter_contract.to_dict()
        return payload


def semantic_fingerprint(program: Any) -> str:
    """Hash executable circuit semantics while excluding transport provenance."""

    ir = ensure_circuit_ir(program)
    interop = ir.metadata.get("interop", {})
    global_phase = (
        interop.get("global_phase", 0.0) if isinstance(interop, dict) else 0.0
    )
    return framework_neutral_fingerprint(
        ir,
        semantic_metadata={
            "global_phase": 0.0 if global_phase is None else global_phase
        },
    )


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
    source_cases = _cases()
    for name, source in source_cases:
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
    barrier = qiskit.QuantumCircuit(1)
    barrier.barrier(0)
    adapter_contract = run_adapter_conformance(
        QISKIT_ADAPTER,
        tuple(
            InteropRoundTripCase(name, source.to_ir()) for name, source in source_cases
        ),
        rejection_cases=(
            InteropRejectionCase(
                "barrier_requires_explicit_lossy_import",
                "import",
                barrier,
                "barrier_dropped",
            ),
        ),
        fingerprint=semantic_fingerprint,
    )
    return QiskitConformanceResult(
        schema="flagquantum_qiskit_conformance_v1",
        qiskit_version=str(qiskit.__version__),
        adapter_contract=adapter_contract,
        cases=cases,
        passed=adapter_contract.passed and all(case.passed for case in cases),
    )


__all__ = (
    "QiskitConformanceCaseResult",
    "QiskitConformanceResult",
    "qiskit_statevector_to_flagquantum",
    "run_qiskit_conformance",
    "semantic_fingerprint",
)
