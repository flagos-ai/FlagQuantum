"""Remote package identity and provider-neutral target profiles."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module as _import_namespace
from typing import Any

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.artifacts.routing_evidence import (
    DEPLOYMENT_PACKAGE_SCHEMA,
    build_deployment_routing_evidence,
    deployment_artifact_sha256,
    stable_payload_sha256,
)
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.core.ir import (
    CircuitIR,
)


@dataclass(frozen=True, init=False)
class CloudBackendProfile:
    """Provider-neutral description of a quantum-cloud target."""

    provider: str
    name: str
    n_qubits: int
    basis_gates: tuple[str, ...] = ()
    coupling_map: CouplingMap | None = None
    supports_openqasm: bool = True
    supports_qcis: bool = False
    supports_dynamic_circuits: bool = False
    max_classical_bits: int | None = None
    is_simulator: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dynamic_dialect: str | None = None

    def __init__(
        self,
        provider: str,
        name: str,
        n_qubits: int | None = None,
        basis_gates: tuple[str, ...] = (),
        coupling_map: CouplingMap | None = None,
        supports_openqasm: bool = True,
        supports_qcis: bool = False,
        supports_dynamic_circuits: bool = False,
        max_classical_bits: int | None = None,
        is_simulator: bool = False,
        metadata: Mapping[str, Any] | None = None,
        dynamic_dialect: str | None = None,
        *,
        n_wires: int | None = None,
    ) -> None:
        if n_qubits is not None and n_wires is not None:
            raise TypeError("Pass n_qubits or n_qubits, not both")
        if n_qubits is None:
            if n_wires is None:
                raise TypeError("CloudBackendProfile requires n_qubits")
            warnings.warn(
                "CloudBackendProfile(n_qubits=...) is deprecated; use "
                "n_qubits=.... Removal is planned for 0.4.0 after the 0.3.x "
                "migration window.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_qubits = n_wires
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "basis_gates", basis_gates)
        object.__setattr__(self, "coupling_map", coupling_map)
        object.__setattr__(self, "supports_openqasm", supports_openqasm)
        object.__setattr__(self, "supports_qcis", supports_qcis)
        object.__setattr__(self, "supports_dynamic_circuits", supports_dynamic_circuits)
        object.__setattr__(self, "max_classical_bits", max_classical_bits)
        object.__setattr__(self, "is_simulator", is_simulator)
        object.__setattr__(self, "metadata", {} if metadata is None else metadata)
        object.__setattr__(self, "dynamic_dialect", dynamic_dialect)

    @property
    def n_wires(self) -> int:
        """Deprecated compatibility alias for :attr:`n_qubits`."""

        return self.n_qubits

    @classmethod
    def simulator(
        cls,
        n_qubits: int | None = None,
        *,
        provider: str = "local",
        n_wires: int | None = None,
    ) -> "CloudBackendProfile":
        if n_qubits is not None and n_wires is not None:
            raise TypeError("Pass n_qubits or n_qubits, not both")
        if n_qubits is None:
            if n_wires is None:
                raise TypeError("CloudBackendProfile.simulator requires n_qubits")
            warnings.warn(
                "CloudBackendProfile.simulator(n_qubits=...) is deprecated; use "
                "n_qubits=.... Removal is planned for 0.4.0 after the 0.3.x "
                "migration window.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_qubits = n_wires
        return cls(
            provider=provider,
            name="simulator",
            n_qubits=int(n_qubits),
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
    def n_qubits(self) -> int:
        return int(self.ir.n_wires)


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
        n_qubits=package.n_qubits,
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
        from flagquantum.runtime.dynamic import (
            DynamicCircuit,
            export_dynamic_qasm3_for_backend,
        )

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


for _identity in (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentPackageIdentityError,
    _native_program,
    validate_deployment_package,
):
    _identity.__module__ = "flagquantum.deployment.cloud"
del _identity

_namespace = _import_namespace("flagquantum.deployment.cloud")
for _name in (
    "Any",
    "CircuitIR",
    "CloudBackendProfile",
    "CouplingMap",
    "DeploymentPackage",
    "DeploymentPackageIdentityError",
    "Mapping",
    "_native_program",
    "validate_deployment_package",
):
    setattr(_namespace, _name, globals()[_name])
del _namespace, _name
