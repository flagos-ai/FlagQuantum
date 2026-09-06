from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic import (
    create_dynamic_deployment_package,
    deploy_dynamic_circuit,
    export_dynamic_qasm3,
    route_dynamic_circuit,
)

pytestmark = pytest.mark.integration


def test_mid_circuit_measurement_collapses_and_controls_a_later_gate() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0, equals=1)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=64, seed=7)

    assert result.samples.shape == (64, 2)
    assert result.classical_bits.shape == (64, 1)
    assert torch.equal(result.samples[:, 0], result.classical_bits[:, 0])
    assert torch.equal(result.samples[:, 1], result.classical_bits[:, 0])
    assert set(result.classical_bits[:, 0].tolist()) == {0, 1}
    assert result.execution_semantics == "local_statevector_trajectory"


def test_reset_returns_measured_qubit_to_zero() -> None:
    circuit = DynamicCircuit(1)
    circuit.h(0)
    circuit.reset(0)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=32, seed=11)

    assert torch.count_nonzero(result.samples) == 0
    assert result.classical_bits.shape == (32, 0)


def test_dynamic_execution_rejects_unmeasured_classical_reads_and_gradients() -> None:
    unread = DynamicCircuit(1)
    unread.conditional("x", 0, classical_bit=0)
    with pytest.raises(RuntimeError, match="read before measurement"):
        fq.experimental.dynamic.run_dynamic(unread, shots=1, seed=1)

    theta = torch.tensor(0.2, requires_grad=True)
    differentiable = DynamicCircuit(1).rx(0, theta=theta)
    differentiable.measure(0, classical_bit=0)
    with pytest.raises(RuntimeError, match="not differentiable"):
        fq.experimental.dynamic.run_dynamic(differentiable, shots=1, seed=1)

    with pytest.raises(fqe.CapabilityError, match="run_dynamic"):
        differentiable.state()
    with pytest.raises(NotImplementedError, match="experimental.dynamic.run_dynamic"):
        fq.run(differentiable)
    with pytest.raises(ValueError, match="outside the circuit"):
        DynamicCircuit(1).measure(1)


def test_dynamic_ir_round_trip_and_qasm3_export() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=1)
    circuit.reset(0)
    circuit.conditional("x", 1, classical_bit=1)

    restored = fq.CircuitIR.from_json(circuit.to_ir().to_json())
    qasm = export_dynamic_qasm3(circuit)

    assert tuple(item.name for item in restored.instructions) == (
        "h",
        "measure",
        "reset",
        "x",
    )
    assert restored.instructions[1].metadata["is_dynamic"] is True
    assert "bit[2] c;" in qasm
    assert "c[1] = measure q[0];" in qasm
    assert "reset q[0];" in qasm
    assert "if (c[1] == true) { x q[1]; }" in qasm


def test_dynamic_execution_supports_batched_initial_states() -> None:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.complex64,
    )
    circuit = DynamicCircuit(1, bsz=2, inputs=inputs)
    circuit.measure(0, classical_bit=0)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=5, seed=3)

    assert result.samples.shape == (2, 5, 1)
    assert result.classical_bits.shape == (2, 5, 1)
    assert result.final_states.shape == (2, 5, 2)
    assert torch.count_nonzero(result.samples[0]) == 0
    assert torch.all(result.samples[1] == 1)
    assert torch.equal(result.samples, result.classical_bits)


def test_multiple_classical_conditions_are_conjoined() -> None:
    circuit = DynamicCircuit(3)
    circuit.x(0).x(1)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=1)
    circuit.conditional("x", 2, conditions={0: 1, 1: 1})

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=4, seed=5)
    qasm = export_dynamic_qasm3(circuit)

    assert torch.all(result.samples == 1)
    assert "if (c[0] == true && c[1] == true) { x q[2]; }" in qasm


def test_dynamic_backend_capability_negotiation_fails_closed() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=2)

    unsupported = fq.CloudBackendProfile.simulator(2)
    rejected = fq.experimental.dynamic.assess_dynamic_backend(circuit, unsupported)
    assert not rejected.compatible
    assert rejected.required_classical_bits == 3
    assert "backend_does_not_declare_dynamic_circuit_support" in rejected.blockers

    supported = fq.CloudBackendProfile(
        provider="test",
        name="dynamic-qpu",
        n_wires=2,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=4,
    )
    accepted = fq.experimental.dynamic.assess_dynamic_backend(circuit, supported)
    assert accepted.compatible
    assert accepted.blockers == ()

    too_small = fq.CloudBackendProfile(
        provider="test",
        name="small-register-qpu",
        n_wires=2,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=2,
    )
    rejected = fq.experimental.dynamic.assess_dynamic_backend(circuit, too_small)
    assert rejected.blockers == ("required_classical_bits_exceed_backend_limit",)


def _dynamic_backend(**overrides) -> fq.CloudBackendProfile:
    values = {
        "provider": "test",
        "name": "dynamic-qpu",
        "n_wires": 4,
        "supports_openqasm": True,
        "supports_dynamic_circuits": True,
        "max_classical_bits": 4,
    }
    values.update(overrides)
    return fq.CloudBackendProfile(**values)


def test_dynamic_deployment_package_is_sealed_and_provider_submittable() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    backend = _dynamic_backend()

    package = create_dynamic_deployment_package(
        circuit,
        backend=backend,
        name="feedback",
        shots=256,
        metadata={"experiment": "bell-feedback"},
    )

    assert fq.deployment.validate_deployment_package(package) is package
    assert package.qasm_version == 3.0
    assert package.shots == 256
    assert package.metadata["dynamic_circuit"] is True
    assert package.metadata["deployment_program_format"] == "openqasm-3"
    assert package.metadata["dynamic_backend_compatibility"]["compatible"] is True
    assert package.metadata["experiment"] == "bell-feedback"

    tampered = replace(package, qasm=package.qasm + "// tampered\n")
    with pytest.raises(
        fq.deployment.DeploymentPackageIdentityError,
        match="QASM",
    ):
        fq.deployment.validate_deployment_package(tampered)

    class RecordingProvider:
        def __init__(self):
            self.package = None

        def run(self, submitted):
            fq.deployment.validate_deployment_package(submitted)
            self.package = submitted
            return {"task_id": "dynamic-1", "shots": submitted.shots}

    provider = RecordingProvider()
    result = deploy_dynamic_circuit(
        circuit,
        provider,
        backend=backend,
        shots=128,
    )
    assert result == {"task_id": "dynamic-1", "shots": 128}
    assert provider.package.metadata["dynamic_circuit"] is True


def test_dynamic_deployment_rejects_unsupported_capacity_and_batching() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)

    with pytest.raises(RuntimeError, match="qubit_capacity"):
        create_dynamic_deployment_package(
            circuit,
            backend=_dynamic_backend(n_wires=1),
        )
    batched = DynamicCircuit(1, bsz=2)
    batched.measure(0, classical_bit=0)
    with pytest.raises(ValueError, match="one circuit"):
        create_dynamic_deployment_package(
            batched,
            backend=_dynamic_backend(),
        )


def test_dynamic_routing_preserves_measurement_and_conditional_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = DynamicCircuit(3)
    circuit.x(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("cx", (0, 2), classical_bit=0)
    circuit.reset(0)
    coupling = fq.CouplingMap.line(3)

    routed = route_dynamic_circuit(circuit, coupling)

    def reject_rerouting(*args: object, **kwargs: object) -> None:
        raise AssertionError("dynamic execution must consume the routed circuit")

    monkeypatch.setattr(
        "flagquantum.runtime.dynamic.routing.route_to_topology",
        reject_rerouting,
    )
    original_result = fq.experimental.dynamic.run_dynamic(circuit, shots=8, seed=17)
    routed_result = fq.experimental.dynamic.run_dynamic(routed, shots=8, seed=17)

    assert torch.equal(original_result.samples, routed_result.samples)
    assert torch.equal(original_result.classical_bits, routed_result.classical_bits)
    assert tuple(item.name for item in routed._instructions) == (
        "x",
        "measure",
        "swap",
        "cx",
        "swap",
        "reset",
    )
    routing = routed.to_ir().metadata["routing"]
    assert routing["mapping_restored"] is True
    assert routing["dynamic_boundary_count"] == 2
    assert routing["dynamic_boundary_source_indices"] == (1, 3)
    assert routing["dynamic_boundary_mapping_policy"] == "identity_restored_per_gate"
    assert routing["conditional_routing_semantics"] == ("unconditional_swap_sandwich")


def test_dynamic_deployment_routes_to_backend_topology_and_seals_evidence() -> None:
    circuit = DynamicCircuit(3)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("cx", (0, 2), classical_bit=0)
    backend = _dynamic_backend(
        n_wires=3,
        coupling_map=fq.CouplingMap.line(3),
    )

    report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)
    package = create_dynamic_deployment_package(
        circuit,
        backend=backend,
        shots=32,
    )

    assert report.compatible
    assert fq.deployment.validate_deployment_package(package) is package
    assert package.metadata["compiled"] is True
    routing = package.metadata["routing_plan"]
    assert routing["inserted_swap_count"] == 2
    assert routing["dynamic_boundary_mapping_policy"] == "identity_restored_per_gate"
    assert "swap q[0], q[1];" in package.qasm


def test_dynamic_deployment_reuses_matching_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("cx", (0, 2), classical_bit=0)
    coupling = fq.CouplingMap.line(3)
    routed = route_dynamic_circuit(circuit, coupling)
    backend = _dynamic_backend(n_wires=3, coupling_map=coupling)

    def reject_rerouting(*args: object, **kwargs: object) -> None:
        raise AssertionError("matching dynamic routing must be reused")

    monkeypatch.setattr(
        "flagquantum.runtime.dynamic.deployment.route_dynamic_circuit",
        reject_rerouting,
    )
    package = create_dynamic_deployment_package(routed, backend=backend, shots=32)

    assert package.metadata["routing_reused"] is True
    assert package.metadata["routing_evidence"]["routing_reused"] is True
    assert package.ir.instructions == routed.to_ir().instructions
