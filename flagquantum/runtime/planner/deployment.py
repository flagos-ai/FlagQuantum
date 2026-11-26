"""Existing remote deployment planning workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import import_module as _import_namespace
from typing import TYPE_CHECKING, Any

from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian
from flagquantum.circuit import Circuit
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler.artifacts.routing_evidence import (
    DEPLOYMENT_PACKAGE_SCHEMA,
    build_deployment_routing_evidence,
    deployment_artifact_sha256,
    same_undirected_device,
    stable_payload_sha256,
)
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.core.ir import (
    CircuitIR,
    Instruction,
    MeasurementNode,
    _reject_native_circuit_input,
)
from flagquantum.runtime.parallel import group_observables

if TYPE_CHECKING:
    from flagquantum.compute.remote.qpu.contracts import (
        DeploymentResult,
        QuantumProvider,
    )

from flagquantum.compute.remote.packages import CloudBackendProfile, DeploymentPackage
from flagquantum.runtime.results.counts import PauliMeasurementPlan


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
    _reject_native_circuit_input(circuit_or_ir)
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
                provider=provider, name=backend_name, n_qubits=ir.n_wires
            )
        else:
            backend = CloudBackendProfile.simulator(ir.n_wires)
    if ir.n_wires > backend.n_qubits:
        raise ValueError(
            "Circuit uses more qubits than the deployment backend exposes."
        )
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
            raise ValueError("compiled target must map every logical qubit")
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
            and same_undirected_device(
                existing_routing.get("coupling_edges", ()),
                backend.coupling_map.edges,
            )
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
        n_qubits=compiled_ir.n_wires,
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

    _reject_native_circuit_input(circuit_or_ir)
    source_ir = (
        circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    )
    if not isinstance(source_ir, CircuitIR):
        raise TypeError("Pauli measurement planning requires Circuit or CircuitIR")
    if hamiltonian.n_qubits > source_ir.n_wires:
        raise ValueError("Hamiltonian references qubits outside the source circuit")
    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    groups = group_observables(hamiltonian.terms)
    packages: list[DeploymentPackage] = []
    all_qubits = tuple(range(source_ir.n_wires))
    for group_index, group in enumerate(groups):
        rotations: list[Instruction] = []
        for qubit, basis in group.basis:
            if basis == "x":
                rotations.append(Instruction("h", (qubit,)))
            elif basis == "y":
                rotations.extend(
                    (
                        Instruction("rz", (qubit,), params={"theta": -math.pi / 2}),
                        Instruction("h", (qubit,)),
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
            measurements=(MeasurementNode("sample", all_qubits, shots=int(shots)),),
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


for _identity in (
    create_deployment_package,
    create_pauli_measurement_plan,
    deploy_circuit,
):
    _identity.__module__ = "flagquantum.deployment.cloud"
del _identity

_namespace = _import_namespace("flagquantum.deployment.cloud")
for _name in (
    "Any",
    "CircuitIR",
    "CloudBackendProfile",
    "DeploymentPackage",
    "Hamiltonian",
    "Mapping",
    "PauliMeasurementPlan",
    "Sequence",
    "create_deployment_package",
    "create_pauli_measurement_plan",
    "deploy_circuit",
):
    setattr(_namespace, _name, globals()[_name])
del _namespace, _name
