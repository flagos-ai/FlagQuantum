"""Executable semantic certification for the optional Qiskit boundary."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import torch

from ...circuit import Circuit
from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
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

_DIFFERENTIAL_SEEDS = (731, 946, 1212, 1597, 2018, 2371)
_QISKIT_OPERATION_NAMES = {"phase": "p", "u3": "u", "cphase": "cp"}
_PARAMETER_NAMES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "u3": ("theta", "phi", "lbd"),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}
_OPERATION_CATALOG = (
    ("x", 1),
    ("y", 1),
    ("z", 1),
    ("h", 1),
    ("s", 1),
    ("sdg", 1),
    ("t", 1),
    ("tdg", 1),
    ("sx", 1),
    ("sxdg", 1),
    ("rx", 1),
    ("ry", 1),
    ("rz", 1),
    ("phase", 1),
    ("u3", 1),
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
    matrix: torch.Tensor | None = None


@dataclass(frozen=True)
class _DifferentialProgram:
    seed: int
    n_wires: int
    operations: tuple[_DifferentialOperation, ...]


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
    ir = replace(
        ir,
        instructions=tuple(
            replace(
                instruction,
                metadata={
                    key: value
                    for key, value in instruction.metadata.items()
                    if key not in {"interop_source", "qiskit_label"}
                },
            )
            for instruction in ir.instructions
        ),
    )
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


def _seeded_unitary(rng: random.Random, width: int) -> torch.Tensor:
    """Build a deterministic asymmetric unitary without a numerical factorization."""

    dimension = 2**width
    permutation = list(range(dimension))
    rng.shuffle(permutation)
    matrix = torch.zeros((dimension, dimension), dtype=torch.complex128)
    for column, row in enumerate(permutation):
        angle = rng.uniform(-math.pi, math.pi)
        matrix[row, column] = complex(math.cos(angle), math.sin(angle))
    return matrix


def _seeded_program(seed: int) -> _DifferentialProgram:
    rng = random.Random(seed)
    n_wires = 3 + seed % 3
    operations: list[_DifferentialOperation] = []
    for wire in range(n_wires):
        operations.append(
            _DifferentialOperation(
                "ry",
                (wire,),
                (rng.uniform(-math.pi, math.pi),),
            )
        )
    for _ in range(12):
        name, arity = rng.choice(_OPERATION_CATALOG)
        wires = tuple(rng.sample(range(n_wires), arity))
        parameters = tuple(
            rng.uniform(-math.pi, math.pi) for _ in _PARAMETER_NAMES.get(name, ())
        )
        operations.append(_DifferentialOperation(name, wires, parameters))
    unitary_width = 2 + seed % 2
    operations.append(
        _DifferentialOperation(
            "unitary",
            tuple(rng.sample(range(n_wires), unitary_width)),
            matrix=_seeded_unitary(rng, unitary_width),
        )
    )
    return _DifferentialProgram(seed, n_wires, tuple(operations))


def _flagquantum_program(program: _DifferentialProgram) -> CircuitIR:
    instructions = []
    for operation in program.operations:
        parameter_names = _PARAMETER_NAMES.get(operation.name, ())
        instructions.append(
            Instruction(
                operation.name,
                operation.wires,
                params=dict(zip(parameter_names, operation.parameters, strict=True)),
                matrix=operation.matrix,
            )
        )
    return CircuitIR(
        program.n_wires,
        tuple(instructions),
        dtype="complex128",
        shape=(1, 2**program.n_wires),
    )


def _qiskit_matrix(matrix: torch.Tensor, width: int) -> Any:
    """Independently convert a wire-major matrix to Qiskit's local bit order."""

    indices = tuple(int(f"{index:0{width}b}"[::-1], 2) for index in range(2**width))
    return matrix[list(indices)][:, list(indices)].numpy()


def _qiskit_program(
    program: _DifferentialProgram,
    *,
    quantum_circuit_type: type[Any],
    unitary_gate_type: type[Any],
) -> Any:
    circuit = quantum_circuit_type(program.n_wires)
    for operation in program.operations:
        if operation.matrix is not None:
            circuit.append(
                unitary_gate_type(
                    _qiskit_matrix(operation.matrix, len(operation.wires)),
                    label=f"seed-{program.seed}",
                ),
                list(operation.wires),
            )
            continue
        method = getattr(
            circuit,
            _QISKIT_OPERATION_NAMES.get(operation.name, operation.name),
        )
        method(*operation.parameters, *operation.wires)
    return circuit


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
            "controlled_rotations",
            Circuit(3, dtype=torch.complex128)
            .gate("h", 1)
            .gate("crx", (1, 2), theta=0.417)
            .gate("cphase", (2, 0), theta=-0.193)
            .gate("swap", (0, 1)),
        ),
        (
            "three_qubit_controls",
            Circuit(3, dtype=torch.complex128)
            .gate("x", 0)
            .gate("x", 1)
            .gate("ccx", (0, 1, 2))
            .gate("cswap", (2, 0, 1)),
        ),
    )


def run_qiskit_conformance(*, atol: float = 1e-10) -> QiskitConformanceResult:
    """Execute deterministic statevector and IR round-trip conformance cases."""

    try:
        qiskit = import_module("qiskit")
        statevector_type = import_module("qiskit.quantum_info").Statevector
        unitary_gate_type = import_module("qiskit.circuit.library").UnitaryGate
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
    seeded_source_cases: list[tuple[str, CircuitIR]] = []
    for seed in _DIFFERENTIAL_SEEDS:
        program = _seeded_program(seed)
        name = f"seeded_differential_{seed}"
        source_ir = _flagquantum_program(program)
        direct_qiskit = _qiskit_program(
            program,
            quantum_circuit_type=qiskit.QuantumCircuit,
            unitary_gate_type=unitary_gate_type,
        )
        exported_qiskit = to_qiskit(source_ir)
        imported_ir = from_qiskit(direct_qiskit)
        round_trip_ir = from_qiskit(exported_qiskit)

        source_state = Circuit.from_ir(source_ir, dtype=torch.complex128).state()[0]
        imported_state = Circuit.from_ir(imported_ir, dtype=torch.complex128).state()[0]
        direct_qiskit_state = qiskit_statevector_to_flagquantum(
            statevector_type.from_instruction(direct_qiskit).data,
            program.n_wires,
        ).to(dtype=torch.complex128)
        exported_qiskit_state = qiskit_statevector_to_flagquantum(
            statevector_type.from_instruction(exported_qiskit).data,
            program.n_wires,
        ).to(dtype=torch.complex128)
        maximum_error = max(
            float(torch.max(torch.abs(source_state - direct_qiskit_state)).item()),
            float(torch.max(torch.abs(source_state - exported_qiskit_state)).item()),
            float(torch.max(torch.abs(imported_state - direct_qiskit_state)).item()),
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
        seeded_source_cases.append((name, source_ir))
    cases = tuple(results)
    barrier = qiskit.QuantumCircuit(1)
    barrier.barrier(0)
    adapter_contract = run_adapter_conformance(
        QISKIT_ADAPTER,
        tuple(
            InteropRoundTripCase(name, source.to_ir()) for name, source in source_cases
        )
        + tuple(
            InteropRoundTripCase(name, source) for name, source in seeded_source_cases
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
