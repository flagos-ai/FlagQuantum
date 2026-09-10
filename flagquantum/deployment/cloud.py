"""Provider-neutral quantum deployment primitives.

This layer turns a trained FlagQuantum circuit into a portable deployment
asset. Real quantum-cloud SDKs can implement ``QuantumProvider`` while the
training, compilation, and algorithm layers keep using the same FlagQuantum
Circuit and IR objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..algorithms import Hamiltonian
from ..circuit import Circuit
from ..compiler import CouplingMap
from ..compiler import compile as compile_program
from ..compiler.openqasm import emit_openqasm
from ..compiler.qcis import emit_qcis
from ..core.ir import CircuitIR, Instruction, MeasurementNode
from ..runtime.parallel import ObservableGroup, group_observables
from .routing_evidence import (
    DEPLOYMENT_PACKAGE_SCHEMA,
    build_deployment_routing_evidence,
    deployment_artifact_sha256,
    stable_payload_sha256,
)

if TYPE_CHECKING:
    from ..remote.qpu.contracts import DeploymentResult, QuantumProvider


@dataclass(frozen=True)
class CloudBackendProfile:
    """Provider-neutral description of a quantum-cloud target."""

    provider: str
    name: str
    n_wires: int
    basis_gates: tuple[str, ...] = ()
    coupling_map: CouplingMap | None = None
    supports_openqasm: bool = True
    supports_qcis: bool = False
    supports_dynamic_circuits: bool = False
    max_classical_bits: int | None = None
    is_simulator: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dynamic_dialect: str | None = None

    @classmethod
    def simulator(
        cls, n_wires: int, *, provider: str = "local"
    ) -> "CloudBackendProfile":
        return cls(
            provider=provider,
            name="simulator",
            n_wires=int(n_wires),
            basis_gates=("x", "y", "z", "h", "rx", "ry", "rz", "cx", "cz", "swap"),
            coupling_map=None,
            is_simulator=True,
        )


@dataclass(frozen=True)
class DeploymentPackage:
    """Portable circuit artifact ready for cloud submission."""

    name: str
    ir: CircuitIR
    qasm: str
    shots: int
    qasm_version: float
    backend: CloudBackendProfile
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_wires(self) -> int:
        return int(self.ir.n_wires)


@dataclass(frozen=True)
class PauliMeasurementPlan:
    """Grouped, basis-rotated deployment packages for one Hamiltonian."""

    hamiltonian: Hamiltonian
    groups: tuple[ObservableGroup, ...]
    packages: tuple[DeploymentPackage, ...]

    def __post_init__(self) -> None:
        if not self.groups or len(self.groups) != len(self.packages):
            raise ValueError("Pauli measurement groups and packages must align")

    @property
    def shots_per_group(self) -> tuple[int, ...]:
        return tuple(package.shots for package in self.packages)

    def expectation(
        self,
        counts_by_group: Sequence[Mapping[str, int]],
    ) -> torch.Tensor:
        return hamiltonian_expectation_from_grouped_counts(counts_by_group, self)

    def standard_error(
        self,
        counts_by_group: Sequence[Mapping[str, int]],
    ) -> torch.Tensor:
        """Return the standard error of the grouped expectation estimator."""

        _, standard_error = _grouped_hamiltonian_statistics(counts_by_group, self)
        return torch.tensor([standard_error], dtype=torch.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "group_count": len(self.groups),
            "term_count": self.hamiltonian.n_terms,
            "shots_per_group": self.shots_per_group,
            "groups": tuple(
                {
                    "term_indices": group.term_indices,
                    "basis": group.basis,
                    "package_name": package.name,
                }
                for group, package in zip(self.groups, self.packages, strict=True)
            ),
        }


class DeploymentPackageIdentityError(ValueError):
    """Raised when a deployment package no longer matches its sealed identity."""


def _native_program(package: DeploymentPackage) -> tuple[str, str]:
    if package.backend.supports_qcis and not package.backend.supports_openqasm:
        qcis = package.metadata.get("qcis")
        if not isinstance(qcis, str) or not qcis.strip():
            raise DeploymentPackageIdentityError(
                "QCIS-native deployment package is missing its QCIS program"
            )
        return "qcis", qcis
    return f"openqasm-{package.qasm_version:g}", package.qasm


def validate_deployment_package(
    package: DeploymentPackage,
) -> DeploymentPackage:
    """Fail closed when provider-facing package identity has been altered."""

    metadata = package.metadata
    if metadata.get("deployment_package_schema") != DEPLOYMENT_PACKAGE_SCHEMA:
        raise DeploymentPackageIdentityError("unsupported deployment package schema")
    if (
        metadata.get("target_provider") != package.backend.provider
        or metadata.get("target_backend") != package.backend.name
    ):
        raise DeploymentPackageIdentityError(
            "deployment package target metadata does not match its backend"
        )
    evidence = metadata.get("routing_evidence")
    if not isinstance(evidence, Mapping):
        raise DeploymentPackageIdentityError("deployment routing evidence is missing")
    if evidence.get("schema") != "flagquantum_deployment_routing_evidence_v1":
        raise DeploymentPackageIdentityError(
            "unsupported deployment routing evidence schema"
        )
    routing_plan = evidence.get("routing_plan")
    if not isinstance(routing_plan, Mapping):
        raise DeploymentPackageIdentityError("deployment routing plan is missing")
    routing_reused = evidence.get("routing_reused")
    if not isinstance(routing_reused, bool):
        raise DeploymentPackageIdentityError(
            "deployment routing reuse marker must be boolean"
        )
    validated_plan = build_deployment_routing_evidence(
        routing_plan,
        routing_reused=routing_reused,
        n_wires=package.n_wires,
        coupling_map=package.backend.coupling_map,
    )
    if dict(evidence) != validated_plan:
        raise DeploymentPackageIdentityError(
            "deployment routing evidence is inconsistent"
        )
    if metadata.get("routing_plan") != routing_plan or metadata.get(
        "routing_reused"
    ) != evidence.get("routing_reused"):
        raise DeploymentPackageIdentityError(
            "deployment routing metadata disagrees with its evidence"
        )
    evidence_digest = stable_payload_sha256(evidence)
    if metadata.get("routing_evidence_sha256") != evidence_digest:
        raise DeploymentPackageIdentityError(
            "deployment routing evidence digest mismatch"
        )
    if metadata.get("dynamic_circuit") is True:
        from ..runtime.dynamic import DynamicCircuit, export_dynamic_qasm3_for_backend

        dynamic = DynamicCircuit(package.ir.n_wires)
        dynamic._instructions.extend(package.ir.instructions)
        expected_qasm = export_dynamic_qasm3_for_backend(dynamic, package.backend)
    else:
        expected_qasm = emit_openqasm(package.ir, version=package.qasm_version)
    if package.qasm != expected_qasm:
        raise DeploymentPackageIdentityError(
            "deployment QASM does not match the packaged IR"
        )
    program_format, program = _native_program(package)
    if metadata.get("deployment_program_format") != program_format:
        raise DeploymentPackageIdentityError(
            "deployment program format does not match its backend"
        )
    expected_artifact = deployment_artifact_sha256(
        name=package.name,
        backend_provider=package.backend.provider,
        backend_name=package.backend.name,
        shots=package.shots,
        program_format=program_format,
        program=program,
        routing_evidence_sha256=evidence_digest,
    )
    if metadata.get("deployment_artifact_sha256") != expected_artifact:
        raise DeploymentPackageIdentityError("deployment artifact digest mismatch")
    return package


def create_deployment_package(
    circuit_or_ir: Circuit | CircuitIR,
    *,
    backend: CloudBackendProfile | None = None,
    name: str = "flagquantum_job",
    shots: int = 1024,
    qasm_version: float = 2.0,
    optimize: bool = True,
    routing_strategy: str = "restore_after_each_gate",
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentPackage:
    """Compile and serialize a trained circuit for cloud deployment."""

    if not isinstance(name, str):
        raise TypeError("deployment name must be a string")
    name = name.strip()
    if not name:
        raise ValueError("deployment name must not be empty")
    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    execution_target = ir.metadata.get("execution_target")
    if execution_target is not None and not isinstance(execution_target, Mapping):
        raise ValueError("execution_target metadata must be a mapping")
    if backend is None:
        if execution_target:
            provider = str(execution_target.get("provider", "")).strip()
            backend_name = str(execution_target.get("backend", "")).strip()
            if not provider or not backend_name:
                raise ValueError("execution_target requires provider and backend")
            backend = CloudBackendProfile(
                provider=provider, name=backend_name, n_wires=ir.n_wires
            )
        else:
            backend = CloudBackendProfile.simulator(ir.n_wires)
    if ir.n_wires > backend.n_wires:
        raise ValueError("Circuit uses more wires than the deployment backend exposes.")
    target_bound = bool(
        execution_target
        and execution_target.get("provider") == backend.provider
        and execution_target.get("backend") == backend.name
    )
    if execution_target and not target_bound:
        raise ValueError("compiled execution_target does not match deployment backend")
    required_options: dict[str, Any] | None = None
    if target_bound and execution_target is not None:
        target_qubits = execution_target.get("target_qubits")
        if execution_target.get("compiler", "missing") is not None:
            raise ValueError("compiled Quafu target requires compiler=None")
        if (
            not isinstance(target_qubits, Sequence)
            or isinstance(target_qubits, (str, bytes))
            or len(target_qubits) != ir.n_wires
        ):
            raise ValueError("compiled target must map every logical wire")
        required_options = {
            "compiler": execution_target.get("compiler"),
            "target_qubits": list(target_qubits),
        }
    existing_routing = ir.metadata.get("routing")
    routing_reused = bool(
        target_bound
        or (
            isinstance(existing_routing, Mapping)
            and backend.coupling_map is not None
            and tuple(existing_routing.get("coupling_edges", ()))
            == backend.coupling_map.edges
            and existing_routing.get("mapping_restored") is True
        )
    )
    compiled_ir = (
        ir
        if target_bound
        else compile_program(
            ir,
            coupling_map=None if routing_reused else backend.coupling_map,
            routing_strategy=routing_strategy,
            optimize=optimize,
        )
    )
    qasm = emit_openqasm(compiled_ir, version=qasm_version)
    package_metadata: dict[str, Any] = {
        "source": "flagquantum",
        "compiled": True,
        "target_provider": backend.provider,
        "target_backend": backend.name,
    }
    package_metadata.update(dict(metadata or {}))
    if required_options is not None:
        provider_options = dict(package_metadata.get("provider_options", {}))
        for key, value in required_options.items():
            if key in provider_options and provider_options[key] != value:
                raise ValueError(
                    f"provider_options.{key} conflicts with compiled target"
                )
            provider_options[key] = value
        package_metadata["provider_options"] = provider_options
    routing_plan = dict(compiled_ir.metadata.get("routing", {}) or {})
    strategy_selection = compiled_ir.metadata.get("routing_strategy_selection")
    if strategy_selection:
        routing_plan["strategy_selection"] = strategy_selection
    routing_evidence = build_deployment_routing_evidence(
        routing_plan,
        routing_reused=routing_reused,
        n_wires=compiled_ir.n_wires,
        coupling_map=backend.coupling_map,
    )
    package_metadata["routing_evidence"] = routing_evidence
    package_metadata["routing_plan"] = routing_evidence["routing_plan"]
    package_metadata["routing_reused"] = routing_evidence["routing_reused"]
    routing_evidence_sha256 = stable_payload_sha256(routing_evidence)
    package_metadata["deployment_package_schema"] = DEPLOYMENT_PACKAGE_SCHEMA
    package_metadata["routing_evidence_sha256"] = routing_evidence_sha256
    if backend.supports_qcis and "qcis" not in package_metadata:
        package_metadata["qcis"] = emit_qcis(compiled_ir)
    program_format = (
        "qcis"
        if backend.supports_qcis and not backend.supports_openqasm
        else f"openqasm-{float(qasm_version):g}"
    )
    program = str(package_metadata["qcis"]) if program_format == "qcis" else qasm
    package_metadata["deployment_program_format"] = program_format
    package_metadata["deployment_artifact_sha256"] = deployment_artifact_sha256(
        name=name,
        backend_provider=backend.provider,
        backend_name=backend.name,
        shots=shots,
        program_format=program_format,
        program=program,
        routing_evidence_sha256=routing_evidence_sha256,
    )
    return DeploymentPackage(
        name=name,
        ir=compiled_ir,
        qasm=qasm,
        shots=int(shots),
        qasm_version=float(qasm_version),
        backend=backend,
        metadata=package_metadata,
    )


def create_pauli_measurement_plan(
    circuit_or_ir: Circuit | CircuitIR,
    hamiltonian: Hamiltonian,
    *,
    backend: CloudBackendProfile | None = None,
    name: str = "flagquantum_measurement",
    shots: int = 1024,
    qasm_version: float = 2.0,
    optimize: bool = True,
    routing_strategy: str = "restore_after_each_gate",
    metadata: Mapping[str, Any] | None = None,
) -> PauliMeasurementPlan:
    """Create qubit-wise-commuting Pauli measurement deployment packages."""

    source_ir = (
        circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    )
    if not isinstance(source_ir, CircuitIR):
        raise TypeError("Pauli measurement planning requires Circuit or CircuitIR")
    if hamiltonian.n_wires > source_ir.n_wires:
        raise ValueError("Hamiltonian references wires outside the source circuit")
    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    groups = group_observables(hamiltonian.terms)
    packages: list[DeploymentPackage] = []
    all_wires = tuple(range(source_ir.n_wires))
    for group_index, group in enumerate(groups):
        rotations: list[Instruction] = []
        for wire, basis in group.basis:
            if basis == "x":
                rotations.append(Instruction("h", (wire,)))
            elif basis == "y":
                rotations.extend(
                    (
                        Instruction("rz", (wire,), params={"theta": -math.pi / 2}),
                        Instruction("h", (wire,)),
                    )
                )
            elif basis != "z":
                raise ValueError(f"unsupported Pauli measurement basis {basis!r}")
        group_metadata = {
            **dict(source_ir.metadata),
            "pauli_measurement": {
                "group_index": group_index,
                "term_indices": group.term_indices,
                "basis": group.basis,
            },
        }
        rotated_ir = replace(
            source_ir,
            instructions=source_ir.instructions + tuple(rotations),
            measurements=(MeasurementNode("sample", all_wires, shots=int(shots)),),
            metadata=group_metadata,
        )
        package_metadata = {
            **dict(metadata or {}),
            "pauli_measurement_group": {
                "group_index": group_index,
                "term_indices": group.term_indices,
                "basis": group.basis,
            },
        }
        packages.append(
            create_deployment_package(
                rotated_ir,
                backend=backend,
                name=f"{name}_group_{group_index}",
                shots=shots,
                qasm_version=qasm_version,
                optimize=optimize,
                routing_strategy=routing_strategy,
                metadata=package_metadata,
            )
        )
    return PauliMeasurementPlan(
        hamiltonian=hamiltonian,
        groups=groups,
        packages=tuple(packages),
    )


def deploy_circuit(
    circuit_or_ir: Circuit | CircuitIR,
    provider: QuantumProvider,
    *,
    backend: CloudBackendProfile | None = None,
    name: str = "flagquantum_job",
    shots: int = 1024,
    qasm_version: float = 2.0,
    optimize: bool = True,
    routing_strategy: str = "restore_after_each_gate",
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentResult:
    """Package and execute a circuit through one remote provider.

    Pass target-compiled IR directly.  Call ``create_deployment_package``
    yourself only when the sealed artifact must be inspected, stored, or
    submitted later.
    """

    package = create_deployment_package(
        circuit_or_ir,
        backend=backend,
        name=name,
        shots=shots,
        qasm_version=qasm_version,
        optimize=optimize,
        routing_strategy=routing_strategy,
        metadata=metadata,
    )
    return provider.run(package)


def _validate_count_histogram(counts: Mapping[str, int]) -> tuple[int, int]:
    """Validate an ungrouped histogram and return its wire and shot counts."""

    if not counts:
        raise ValueError("Counts cannot be empty.")
    n_wires = len(str(next(iter(counts))))
    if n_wires == 0:
        raise ValueError("Count bitstrings cannot be empty.")
    shots = 0
    for bitstring, count in counts.items():
        bits = str(bitstring)
        if len(bits) != n_wires:
            raise ValueError("Count bitstrings must have the same width.")
        if any(bit not in "01" for bit in bits):
            raise ValueError("Count bitstrings must be binary.")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("Counts must be non-negative integers.")
        if count < 0:
            raise ValueError("Counts must be non-negative integers.")
        shots += int(count)
    if shots == 0:
        raise ValueError("Counts must contain at least one shot.")
    return n_wires, shots


def expectation_z_from_counts(
    counts: Mapping[str, int],
    wires: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Estimate Z expectations from cloud measurement counts."""

    n_wires, shot_count = _validate_count_histogram(counts)
    if wires is None:
        wire_tuple = tuple(range(n_wires))
    elif isinstance(wires, int):
        wire_tuple = (wires,)
    else:
        wire_tuple = tuple(int(wire) for wire in wires)

    if any(wire < 0 or wire >= n_wires for wire in wire_tuple):
        raise ValueError("Requested wires are outside the measured register.")
    shots = float(shot_count)
    values = []
    for wire in wire_tuple:
        total = 0.0
        for bitstring, count in counts.items():
            bit = int(str(bitstring)[wire])
            total += (1.0 if bit == 0 else -1.0) * int(count)
        values.append(total / shots)
    return torch.tensor([values], dtype=torch.float32)


def hamiltonian_expectation_from_counts(
    counts: Mapping[str, int],
    hamiltonian: Hamiltonian,
) -> torch.Tensor:
    """Estimate an I/Z-only Hamiltonian from computational-basis counts."""

    n_wires, shot_count = _validate_count_histogram(counts)
    if hamiltonian.n_wires > n_wires:
        raise ValueError("Hamiltonian references wires outside the measured register.")
    shots = float(shot_count)
    total = 0.0
    for term in hamiltonian.terms:
        coeff = float(torch.real(torch.as_tensor(term.coefficient)).detach())
        term_total = 0.0
        for bitstring, count in counts.items():
            parity = 1.0
            for wire, name in term.ops:
                if name != "z":
                    raise ValueError(
                        "Counts-based Hamiltonian estimation currently supports only I/Z terms. "
                        "Use basis-rotated deployment packages for X/Y observables."
                    )
                parity *= 1.0 if str(bitstring)[wire] == "0" else -1.0
            term_total += parity * int(count)
        total += coeff * term_total / shots
    return torch.tensor([total], dtype=torch.float32)


def hamiltonian_expectation_from_grouped_counts(
    counts_by_group: Sequence[Mapping[str, int]],
    plan: PauliMeasurementPlan,
) -> torch.Tensor:
    """Aggregate basis-rotated counts from a ``PauliMeasurementPlan``."""

    expectation, _ = _grouped_hamiltonian_statistics(counts_by_group, plan)
    return torch.tensor([expectation], dtype=torch.float32)


def _grouped_hamiltonian_statistics(
    counts_by_group: Sequence[Mapping[str, int]],
    plan: PauliMeasurementPlan,
) -> tuple[float, float]:
    """Return the grouped estimator mean and standard error."""

    if len(counts_by_group) != len(plan.groups):
        raise ValueError(
            "counts_by_group must contain one result per measurement group"
        )
    total = 0.0
    total_variance = 0.0
    for group_index, (counts, group, package) in enumerate(
        zip(counts_by_group, plan.groups, plan.packages, strict=True)
    ):
        if not counts:
            raise ValueError(f"measurement group {group_index} counts cannot be empty")
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError(
                    f"measurement group {group_index} counts must be non-negative integers"
                )
            if count < 0:
                raise ValueError(
                    f"measurement group {group_index} counts must be non-negative integers"
                )
        shots = sum(int(count) for count in counts.values())
        if shots != package.shots:
            raise ValueError(
                f"measurement group {group_index} returned {shots} shots; "
                f"expected {package.shots}"
            )
        for bitstring in counts:
            if len(str(bitstring)) != package.n_wires:
                raise ValueError(
                    f"measurement group {group_index} bitstring width does not "
                    "match its deployment package"
                )
            if any(bit not in "01" for bit in str(bitstring)):
                raise ValueError(
                    f"measurement group {group_index} bitstrings must be binary"
                )
        coefficients: list[tuple[float, tuple[tuple[int, str], ...]]] = []
        for term_index in group.term_indices:
            term = plan.hamiltonian.terms[term_index]
            coefficient_tensor = torch.as_tensor(term.coefficient)
            if (
                coefficient_tensor.is_complex()
                and torch.abs(coefficient_tensor.imag) > 1e-12
            ):
                raise ValueError("measured Hamiltonian coefficients must be real")
            coefficients.append((float(coefficient_tensor.real), term.ops))

        mean = 0.0
        second_moment = 0.0
        for bitstring, count in counts.items():
            sample_value = 0.0
            for coefficient, ops in coefficients:
                parity = 1.0
                for wire, _name in ops:
                    parity *= 1.0 if str(bitstring)[wire] == "0" else -1.0
                sample_value += coefficient * parity
            probability = int(count) / shots
            mean += probability * sample_value
            second_moment += probability * sample_value * sample_value
        total += mean
        total_variance += max(second_moment - mean * mean, 0.0) / shots
    return total, math.sqrt(total_variance)


__all__ = [
    "CloudBackendProfile",
    "DeploymentPackage",
    "DeploymentPackageIdentityError",
    "PauliMeasurementPlan",
    "create_deployment_package",
    "create_pauli_measurement_plan",
    "deploy_circuit",
    "hamiltonian_expectation_from_counts",
    "hamiltonian_expectation_from_grouped_counts",
    "expectation_z_from_counts",
    "validate_deployment_package",
]
