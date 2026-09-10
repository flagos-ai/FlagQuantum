"""Executable complex128 certification for the PennyLane IR boundary."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

from ...circuit import Circuit
from ..conformance import (
    InteropConformanceResult,
    InteropRejectionCase,
    InteropRoundTripCase,
    run_adapter_conformance,
)
from .adapter import PENNYLANE_ADAPTER
from .conversion import from_pennylane, to_pennylane
from .models import PennyLaneDependencyError


@dataclass(frozen=True)
class PennyLaneConformanceCaseResult:
    name: str
    passed: bool
    maximum_absolute_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "maximum_absolute_error": self.maximum_absolute_error,
        }


@dataclass(frozen=True)
class PennyLaneConformanceResult:
    schema: str
    pennylane_version: str
    dtype: str
    cases: tuple[PennyLaneConformanceCaseResult, ...]
    adapter_contract: InteropConformanceResult
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pennylane_version": self.pennylane_version,
            "dtype": self.dtype,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
            "adapter_contract": self.adapter_contract.to_dict(),
        }


def _cases() -> tuple[tuple[str, Circuit], ...]:
    return (
        (
            "asymmetric_wire_order",
            Circuit(3, dtype=torch.complex128)
            .gate("x", 0)
            .gate("ry", 2, theta=0.371)
            .gate("cx", (0, 1))
            .gate("rz", 1, theta=-0.219),
        ),
        (
            "controlled_and_ising",
            Circuit(3, dtype=torch.complex128)
            .gate("h", 1)
            .gate("crx", (1, 2), theta=0.417)
            .gate("rzz", (2, 0), theta=-0.193),
        ),
    )


def run_pennylane_conformance(*, atol: float = 1e-10) -> PennyLaneConformanceResult:
    """Certify static round trips and PennyLane matrices at complex128."""

    try:
        qml = import_module("pennylane")
    except ImportError as exc:
        raise PennyLaneDependencyError(
            "PennyLane conformance requires `pip install 'flagquantum[pennylane]'`."
        ) from exc
    source_cases = _cases()
    results: list[PennyLaneConformanceCaseResult] = []
    for name, source in source_cases:
        source_ir = source.to_ir()
        script = to_pennylane(source_ir)
        round_trip = from_pennylane(script)
        matrix = torch.as_tensor(
            qml.matrix(script, wire_order=list(range(source_ir.n_wires)))
        ).to(torch.complex128)
        initial = torch.zeros(2**source_ir.n_wires, dtype=torch.complex128)
        initial[0] = 1
        pennylane_state = matrix @ initial
        flagquantum_state = Circuit.from_ir(round_trip, dtype=torch.complex128).state()[
            0
        ]
        error = float(torch.max(torch.abs(pennylane_state - flagquantum_state)).item())
        results.append(PennyLaneConformanceCaseResult(name, error <= atol, error))
    unsupported = qml.tape.QuantumScript([qml.Rot(0.1, 0.2, 0.3, 0)])
    adapter_contract = run_adapter_conformance(
        PENNYLANE_ADAPTER,
        tuple(
            InteropRoundTripCase(name, circuit.to_ir())
            for name, circuit in source_cases
        ),
        rejection_cases=(
            InteropRejectionCase(
                "unsupported_rot_requires_explicit_lossy_import",
                "import",
                unsupported,
                "unsupported_operation",
            ),
        ),
    )
    cases = tuple(results)
    return PennyLaneConformanceResult(
        "flagquantum_pennylane_conformance_v1",
        str(qml.__version__),
        "complex128",
        cases,
        adapter_contract,
        adapter_contract.passed and all(case.passed for case in cases),
    )


__all__ = (
    "PennyLaneConformanceCaseResult",
    "PennyLaneConformanceResult",
    "run_pennylane_conformance",
)
