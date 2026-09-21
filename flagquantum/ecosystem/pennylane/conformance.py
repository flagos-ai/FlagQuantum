"""Executable complex128 certification for the PennyLane IR boundary."""

from __future__ import annotations

import math
import random
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

_DIFFERENTIAL_SEEDS = (731, 946, 1212, 1597, 2018, 2371)
_PENNYLANE_OPERATION_NAMES = {
    "i": "Identity",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "h": "Hadamard",
    "s": "S",
    "t": "T",
    "sx": "SX",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "phase": "PhaseShift",
    "cx": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "cphase": "ControlledPhaseShift",
    "rxx": "IsingXX",
    "ryy": "IsingYY",
    "rzz": "IsingZZ",
    "ccx": "Toffoli",
    "cswap": "CSWAP",
}
_PARAMETER_NAMES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}
_OPERATION_CATALOG = (
    ("i", 1),
    ("x", 1),
    ("y", 1),
    ("z", 1),
    ("h", 1),
    ("s", 1),
    ("t", 1),
    ("sx", 1),
    ("rx", 1),
    ("ry", 1),
    ("rz", 1),
    ("phase", 1),
    ("cx", 2),
    ("cy", 2),
    ("cz", 2),
    ("swap", 2),
    ("crx", 2),
    ("cry", 2),
    ("crz", 2),
    ("cphase", 2),
    ("rxx", 2),
    ("ryy", 2),
    ("rzz", 2),
    ("ccx", 3),
    ("cswap", 3),
)


@dataclass(frozen=True)
class _DifferentialOperation:
    name: str
    wires: tuple[int, ...]
    parameters: tuple[float, ...] = ()


@dataclass(frozen=True)
class _DifferentialProgram:
    n_wires: int
    operations: tuple[_DifferentialOperation, ...]


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


def _seeded_program(seed: int) -> _DifferentialProgram:
    rng = random.Random(seed)
    n_wires = 3 + seed % 3
    operations = [
        _DifferentialOperation(
            "ry",
            (wire,),
            (rng.uniform(-math.pi, math.pi),),
        )
        for wire in range(n_wires)
    ]
    for name, arity in (("cx", 2), ("ccx", 3)):
        operations.append(
            _DifferentialOperation(
                name,
                tuple(rng.sample(range(n_wires), arity)),
            )
        )
    for _ in range(10):
        name, arity = rng.choice(_OPERATION_CATALOG)
        parameters = tuple(
            rng.uniform(-math.pi, math.pi) for _ in _PARAMETER_NAMES.get(name, ())
        )
        operations.append(
            _DifferentialOperation(
                name,
                tuple(rng.sample(range(n_wires), arity)),
                parameters,
            )
        )
    return _DifferentialProgram(n_wires, tuple(operations))


def _flagquantum_circuit(program: _DifferentialProgram) -> Circuit:
    circuit = Circuit(program.n_wires, dtype=torch.complex128)
    for operation in program.operations:
        parameters = dict(
            zip(
                _PARAMETER_NAMES.get(operation.name, ()),
                operation.parameters,
                strict=True,
            )
        )
        circuit = circuit.gate(operation.name, operation.wires, params=parameters)
    return circuit


def _pennylane_script(qml: Any, program: _DifferentialProgram) -> Any:
    operations = []
    for operation in program.operations:
        operation_type = getattr(qml, _PENNYLANE_OPERATION_NAMES[operation.name])
        wires: Any = (
            operation.wires[0] if len(operation.wires) == 1 else operation.wires
        )
        operations.append(operation_type(*operation.parameters, wires=wires))
    return qml.tape.QuantumScript(operations, measurements=(), shots=None)


def _pennylane_state(qml: Any, script: Any, n_wires: int) -> torch.Tensor:
    matrix = torch.as_tensor(qml.matrix(script, wire_order=list(range(n_wires)))).to(
        torch.complex128
    )
    initial = torch.zeros(2**n_wires, dtype=torch.complex128)
    initial[0] = 1
    return matrix @ initial


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


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
        pennylane_state = _pennylane_state(qml, script, source_ir.n_wires)
        flagquantum_state = Circuit.from_ir(round_trip, dtype=torch.complex128).state()[
            0
        ]
        error = _maximum_error(pennylane_state, flagquantum_state)
        results.append(PennyLaneConformanceCaseResult(name, error <= atol, error))
    seeded_cases: list[tuple[str, Circuit]] = []
    for seed in _DIFFERENTIAL_SEEDS:
        name = f"seeded_differential_{seed}"
        program = _seeded_program(seed)
        source = _flagquantum_circuit(program)
        source_ir = source.to_ir()
        reference_script = _pennylane_script(qml, program)
        exported_script = to_pennylane(source_ir)
        imported_ir = from_pennylane(reference_script)

        reference_state = _pennylane_state(qml, reference_script, program.n_wires)
        source_state = source.state()[0]
        exported_state = _pennylane_state(qml, exported_script, program.n_wires)
        imported_state = Circuit.from_ir(imported_ir, dtype=torch.complex128).state()[0]
        error = max(
            _maximum_error(source_state, reference_state),
            _maximum_error(source_state, exported_state),
            _maximum_error(imported_state, reference_state),
        )
        results.append(PennyLaneConformanceCaseResult(name, error <= atol, error))
        seeded_cases.append((name, source))
    unsupported = qml.tape.QuantumScript([qml.Rot(0.1, 0.2, 0.3, 0)])
    adapter_contract = run_adapter_conformance(
        PENNYLANE_ADAPTER,
        tuple(
            InteropRoundTripCase(name, circuit.to_ir())
            for name, circuit in (*source_cases, *seeded_cases)
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
