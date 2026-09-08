"""Provider-neutral quantum deployment primitives.

This layer turns a trained FlagQuantum circuit into a portable deployment
asset. Real quantum-cloud SDKs can implement ``QuantumProvider`` while the
training, compilation, and algorithm layers keep using the same FlagQuantum
Circuit and IR objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    if backend is None:
        backend = CloudBackendProfile.simulator(ir.n_wires)
    if ir.n_wires > backend.n_wires:
        raise ValueError("Circuit uses more wires than the deployment backend exposes.")
    existing_routing = ir.metadata.get("routing")
    routing_reused = bool(
        isinstance(existing_routing, Mapping)
        and backend.coupling_map is not None
        and tuple(existing_routing.get("coupling_edges", ()))
        == backend.coupling_map.edges
        and existing_routing.get("mapping_restored") is True
    )
    compiled_ir = compile_program(
        ir,
        coupling_map=None if routing_reused else backend.coupling_map,
        routing_strategy=routing_strategy,
        optimize=optimize,
    )
    qasm = emit_openqasm(compiled_ir, version=qasm_version)
    package_metadata = {
        "source": "flagquantum",
        "compiled": True,
        "target_provider": backend.provider,
        "target_backend": backend.name,
    }
    package_metadata.update(dict(metadata or {}))
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
                    (Instruction("sdg", (wire,)), Instruction("h", (wire,)))
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
    """Package a circuit and run it on a provider."""

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


def expectation_z_from_counts(
    counts: Mapping[str, int],
    wires: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Estimate Z expectations from cloud measurement counts."""

    if not counts:
        raise ValueError("Counts cannot be empty.")
    first_key = next(iter(counts))
    n_wires = len(str(first_key))
    if wires is None:
        wire_tuple = tuple(range(n_wires))
    elif isinstance(wires, int):
        wire_tuple = (wires,)
    else:
        wire_tuple = tuple(int(wire) for wire in wires)

    shots = float(sum(int(value) for value in counts.values()))
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

    if not counts:
        raise ValueError("Counts cannot be empty.")
    shots = float(sum(int(value) for value in counts.values()))
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

    if len(counts_by_group) != len(plan.groups):
        raise ValueError(
            "counts_by_group must contain one result per measurement group"
        )
    total = 0.0
    for group_index, (counts, group, package) in enumerate(
        zip(counts_by_group, plan.groups, plan.packages, strict=True)
    ):
        if not counts:
            raise ValueError(f"measurement group {group_index} counts cannot be empty")
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
        for term_index in group.term_indices:
            term = plan.hamiltonian.terms[term_index]
            coefficient = torch.as_tensor(term.coefficient)
            if coefficient.is_complex() and torch.abs(coefficient.imag) > 1e-12:
                raise ValueError("measured Hamiltonian coefficients must be real")
            expectation = 0.0
            for bitstring, count in counts.items():
                parity = 1.0
                for wire, _name in term.ops:
                    parity *= 1.0 if str(bitstring)[wire] == "0" else -1.0
                expectation += parity * int(count)
            total += float(coefficient.real) * expectation / shots
    return torch.tensor([total], dtype=torch.float32)


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
