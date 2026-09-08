"""Tests for FlagQuantum train-to-cloud deployment primitives."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.deployment as deployment
import flagquantum.deployment as fqd
from flagquantum.algorithms import Hamiltonian, pauli_term, run_vqe
from flagquantum.compiler import CouplingMap
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.deployment import (
    CloudBackendProfile,
    DeploymentPackageIdentityError,
    expectation_z_from_counts,
    hamiltonian_expectation_from_counts,
    validate_deployment_package,
    validate_deployment_result,
)
from flagquantum.testing import InMemoryRemoteTarget


def test_create_deployment_package_exports_qasm_and_metadata():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    package = fqd.create_deployment_package(
        circuit,
        backend=CloudBackendProfile.simulator(2),
        name="bell_inference",
        shots=128,
        qasm_version=2.0,
        metadata={"stage": "inference"},
    )

    assert package.name == "bell_inference"
    assert package.shots == 128
    assert package.n_wires == 2
    assert package.metadata["stage"] == "inference"
    assert "OPENQASM 2.0;" in package.qasm
    assert "cx q[0], q[1];" in package.qasm


def test_deployment_package_uses_backend_topology():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    backend = CloudBackendProfile(
        provider="local",
        name="line3",
        n_wires=3,
        coupling_map=CouplingMap.line(3),
        is_simulator=True,
    )

    package = fqd.create_deployment_package(circuit, backend=backend, shots=16)

    assert package.ir.instructions[1].name == "swap"
    for instruction in package.ir.instructions:
        if len(instruction.wires) == 2:
            assert backend.coupling_map.has_edge(*instruction.wires)


def test_deployment_package_preserves_auto_routing_selection_evidence():
    circuit = fq.Circuit(5)
    circuit.cx(0, 4).h(4).cx(0, 4)
    backend = CloudBackendProfile(
        provider="local",
        name="line5",
        n_wires=5,
        coupling_map=CouplingMap.line(5),
        is_simulator=True,
    )

    package = fqd.create_deployment_package(
        circuit,
        backend=backend,
        routing_strategy="auto",
        metadata={
            "routing_plan": {"strategy": "tampered"},
            "routing_reused": "tampered",
            "routing_evidence": {"schema": "tampered"},
        },
    )

    routing = package.metadata["routing_plan"]
    assert routing["strategy"] == "persistent_layout"
    assert routing["strategy_selection"]["selected_strategy"] == "persistent_layout"
    assert package.metadata["routing_reused"] is False
    assert package.metadata["routing_evidence"]["schema"] == (
        "flagquantum_deployment_routing_evidence_v1"
    )
    assert torch.allclose(fq.Circuit.from_ir(package.ir).state(), circuit.state())


def test_deployment_reuses_compatible_compiled_routing_plan():
    circuit = fq.Circuit(4)
    circuit.x(3).cx(0, 3).h(0).cx(0, 3)
    coupling = CouplingMap.line(4)
    compiled = circuit.compile(
        coupling_map=coupling,
        routing_strategy="persistent_layout",
    )
    backend = CloudBackendProfile(
        provider="local",
        name="line4",
        n_wires=4,
        coupling_map=coupling,
        is_simulator=True,
    )

    package = fqd.create_deployment_package(compiled.to_ir(), backend=backend)

    assert package.metadata["routing_reused"] is True
    assert package.metadata["routing_plan"]["strategy"] == "persistent_layout"
    assert package.ir.instructions == compiled.to_ir().instructions
    assert torch.allclose(
        fq.Circuit.from_ir(package.ir).state(),
        circuit.state(),
        atol=1e-6,
    )


def test_qcis_emitter_uses_native_gate_decomposition():
    theta = torch.tensor(0.25)
    circuit = fq.Circuit(2)
    circuit.h(0).rx(1, theta=theta).cx(0, 1).rzz(0, 1, theta=0.5)

    qcis = emit_qcis(circuit)
    lines = qcis.splitlines()

    assert lines[0] == "Y2M Q0"
    assert "RZ Q1 0.25" in lines
    assert "CZ Q0 Q1" in qcis
    assert all(
        line.split()[0] in {"X2P", "X2M", "Y2P", "Y2M", "RZ", "CZ", "I"}
        for line in lines
    )


def test_qcis_backend_package_gets_qcis_metadata_automatically():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    backend = CloudBackendProfile(
        provider="tianyan",
        name="tianyan176",
        n_wires=8,
        supports_openqasm=False,
        supports_qcis=True,
    )

    package = fqd.create_deployment_package(circuit, backend=backend, shots=32)

    assert package.metadata["target_provider"] == "tianyan"
    assert package.metadata["qcis"].startswith("Y2M Q0")
    assert "CZ Q0 Q1" in package.metadata["qcis"]
    assert package.qasm.startswith("OPENQASM 2.0;")


def test_local_provider_runs_packaged_circuit():
    circuit = fq.Circuit(2)
    circuit.x(0).x(1)
    provider = InMemoryRemoteTarget()
    backend = provider.discover_backends(2)[0]

    result = fqd.deploy_circuit(circuit, provider, backend=backend, shots=32)

    assert result.handle.provider == "local"
    assert (
        result.handle.payload["deployment_artifact_sha256"]
        == result.metadata["deployment_artifact_sha256"]
    )
    assert (
        result.handle.payload["routing_evidence_sha256"]
        == result.metadata["routing_evidence_sha256"]
    )
    assert result.counts == {"11": 32}
    assert torch.allclose(expectation_z_from_counts(result.counts), -torch.ones(1, 2))
    assert torch.allclose(
        hamiltonian_expectation_from_counts(
            result.counts,
            Hamiltonian([pauli_term(0.5, "ZZ", (0, 1))]),
        ),
        torch.tensor([0.5]),
    )


def test_trained_circuit_can_be_packaged_for_inference():
    theta = torch.tensor([0.4, -0.2], requires_grad=True)
    hamiltonian = Hamiltonian(
        [
            pauli_term(1.0, "Z", (0,)),
            pauli_term(0.1, "Z", (1,)),
        ]
    )

    def builder(params):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=params[0])
        circuit.ry(1, theta=params[1])
        circuit.cx(0, 1)
        return circuit

    result = run_vqe(builder, theta, hamiltonian, steps=5, lr=0.1)
    trained_circuit = builder(result.parameters)
    optimized = result.parameters.detach()
    package = fqd.create_deployment_package(
        trained_circuit,
        backend=CloudBackendProfile.simulator(2),
        name="trained_vqe_inference",
        metadata={"optimized_parameters": optimized.tolist()},
    )

    assert package.metadata["compiled"]
    assert package.metadata["optimized_parameters"] == optimized.tolist()
    assert f"rx({float(optimized[0])})" in package.qasm
    assert f"ry({float(optimized[1])})" in package.qasm
    assert package.ir.n_wires == 2


def test_symbolic_parameter_template_binds_optimized_values_for_deployment():
    theta0 = fq.Parameter("theta0")
    theta1 = fq.Parameter("theta1")
    template = fq.Circuit(2)
    template.rx(0, theta=theta0).ry(1, theta=theta1).cx(0, 1)
    optimized = torch.tensor([0.12, -0.34])

    trained_circuit = template.bind_parameters(
        {
            "theta0": optimized[0],
            "theta1": optimized[1],
        }
    )
    backend = CloudBackendProfile(
        provider="guodun",
        name="gd_qc1",
        n_wires=8,
        supports_openqasm=False,
        supports_qcis=True,
    )
    package = fqd.create_deployment_package(
        trained_circuit,
        backend=backend,
        metadata={"optimized_parameters": optimized.tolist()},
    )

    assert not trained_circuit.is_parameterized()
    assert package.metadata["optimized_parameters"] == optimized.tolist()
    assert "rx(0.119999" in package.qasm
    assert "RZ Q0" in package.metadata["qcis"]
    assert "CZ Q0 Q1" in package.metadata["qcis"]


def test_deployment_subsystem_is_top_level_easy_to_use():
    assert deployment.CloudBackendProfile is CloudBackendProfile


@pytest.mark.parametrize("field", ("qasm", "shots", "routing_evidence"))
def test_provider_rejects_tampered_deployment_package(field):
    provider = InMemoryRemoteTarget()
    backend = provider.discover_backends(2)[0]
    package = fqd.create_deployment_package(
        fq.Circuit(2).h(0).cx(0, 1),
        backend=backend,
        shots=8,
    )
    if field == "qasm":
        tampered = replace(package, qasm=package.qasm + "\n// tampered")
    elif field == "shots":
        tampered = replace(package, shots=9)
    else:
        metadata = deepcopy(package.metadata)
        metadata["routing_evidence"]["status"] = "tampered"
        tampered = replace(package, metadata=metadata)

    with pytest.raises(DeploymentPackageIdentityError, match="deployment"):
        provider.submit(tampered)


def test_qcis_native_program_is_bound_to_deployment_identity():
    backend = CloudBackendProfile(
        provider="tianyan",
        name="qpu",
        n_wires=2,
        supports_openqasm=False,
        supports_qcis=True,
    )
    package = fqd.create_deployment_package(
        fq.Circuit(2).h(0).cx(0, 1),
        backend=backend,
    )
    metadata = deepcopy(package.metadata)
    metadata["qcis"] += "\nX Q0"

    assert package.metadata["deployment_program_format"] == "qcis"
    with pytest.raises(DeploymentPackageIdentityError, match="artifact digest"):
        validate_deployment_package(replace(package, metadata=metadata))


@pytest.mark.parametrize("field", ("identity", "shots"))
def test_deployment_result_rejects_broken_receipt_chain(field):
    provider = InMemoryRemoteTarget()
    backend = provider.discover_backends(2)[0]
    result = fqd.deploy_circuit(
        fq.Circuit(2).x(0),
        provider,
        backend=backend,
        shots=8,
    )
    if field == "identity":
        metadata = dict(result.metadata)
        metadata["deployment_artifact_sha256"] = "0" * 64
        tampered = replace(result, metadata=metadata)
    else:
        tampered = replace(result, shots=9)

    with pytest.raises(DeploymentPackageIdentityError, match="deployment"):
        validate_deployment_result(tampered)
