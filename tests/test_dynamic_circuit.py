from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.deployment as deployment
import flagquantum.errors as fqe
from flagquantum.compiler import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.deployment import CloudBackendProfile
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


def test_repeated_measurement_and_reset_on_one_qubit_have_defined_semantics() -> None:
    circuit = DynamicCircuit(1)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.measure(0, classical_bit=1)
    circuit.reset(0)
    circuit.measure(0, classical_bit=2)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=64, seed=19)

    assert torch.equal(result.classical_bits[:, 0], result.classical_bits[:, 1])
    assert torch.count_nonzero(result.classical_bits[:, 2]) == 0
    assert torch.count_nonzero(result.samples) == 0


def test_seeded_dynamic_execution_is_reproducible_across_processes() -> None:
    source = """
        import json

        import flagquantum as fq
        from flagquantum.dynamic import DynamicCircuit

        circuit = DynamicCircuit(2)
        circuit.h(0)
        circuit.measure(0, classical_bit=0)
        circuit.conditional("x", 1, classical_bit=0)
        result = fq.experimental.dynamic.run_dynamic(circuit, shots=64, seed=23)
        print(json.dumps({
            "classical": result.classical_bits.tolist(),
            "samples": result.samples.tolist(),
        }, sort_keys=True))
        """
    program = textwrap.dedent(source)

    outputs = tuple(
        subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(2)
    )

    assert outputs[0] == outputs[1]


def test_local_dynamic_feedback_example_is_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.local.dynamic_feedback"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "local_statevector_trajectory" in completed.stdout


def test_local_dynamic_validation_example_is_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.local.dynamic_validation"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "build-time validation: classical bit 0 was read before measurement" in (
        completed.stdout
    )
    assert "distributed boundary: dynamic_conditional_execution_unsupported" in (
        completed.stdout
    )


def test_dynamic_execution_rejects_unmeasured_classical_reads_and_gradients() -> None:
    with pytest.raises(fqe.ValidationError, match="read before measurement"):
        DynamicCircuit(1).conditional("x", 0, classical_bit=0)

    # A circuit loaded from IR did not pass through the builder, so execution
    # repeats the same validation before allocating trajectories.
    unread = DynamicCircuit.from_ir(
        CircuitIR(
            1,
            (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),),
        )
    )
    with pytest.raises(fqe.ValidationError, match="read before measurement"):
        fq.experimental.dynamic.run_dynamic(unread, shots=1, seed=1)
    with pytest.raises(fqe.CapabilityError, match="run_dynamic"):
        fq.run(unread)

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


def test_disjunctive_condition_runs_the_gate_on_either_conjunction() -> None:
    circuit = (
        DynamicCircuit(3, dtype=torch.complex128)
        .ry(0, theta=0.7)
        .ry(1, theta=1.1)
        .measure(0, classical_bit=0)
        .measure(1, classical_bit=1)
        .conditional("x", 2, any_of=({0: 1}, {1: 1}))
    )

    # The two clauses are a disjunction, so the gate runs when either readout is
    # one. A conjunction would run on the intersection instead, and the exact
    # density matrix separates the two answers.
    populations = circuit.density_matrix()[0].diagonal().real
    ran = float(sum(populations[index] for index in (1, 3, 5, 7)))
    either = 1.0 - math.cos(0.35) ** 2 * math.cos(0.55) ** 2
    assert ran == pytest.approx(either, abs=1e-12)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=20_000, seed=3)
    ran_shots = int(result.final_samples[:, 2].sum())
    assert abs(ran_shots / 20_000 - either) < 0.02


def test_planner_refuses_every_condition_spelling_with_one_message() -> None:
    conjunction = DynamicCircuit(2)
    conjunction.measure(0, classical_bit=0)
    conjunction.conditional("x", 1, classical_bit=0)

    disjunctive = DynamicCircuit(2)
    disjunctive.measure(0, classical_bit=0)
    disjunctive.measure(1, classical_bit=1)
    disjunctive.conditional("x", 1, any_of=({0: 1}, {1: 1}))

    # Every program the builder produces carries a measurement, and a measurement
    # carries is_dynamic, so the two above reach the guard through that key as
    # well. This one reaches it through the condition key alone: a program read
    # from IR need not have been built here, and a measurement is what would
    # otherwise be covering for the condition.
    from_ir = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction(
                "x", (1,), metadata={"condition_clauses": (((0, 1),), ((1, 0),))}
            ),
        ),
    )

    messages = []
    for program in (conjunction, disjunctive, from_ir):
        with pytest.raises(fqe.CapabilityError, match="run_dynamic") as caught:
            fq.plan(program)
        messages.append(str(caught.value))

    # One guard reads every condition spelling, so the three spellings get one
    # explanation rather than one per spelling.
    assert len(set(messages)) == 1


def test_qasm3_export_writes_a_disjunction_of_conjunctions() -> None:
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=1)
    conjunction = DynamicCircuit(3)
    conjunction.measure(0, classical_bit=0)
    conjunction.measure(1, classical_bit=1)
    conjunction.conditional("x", 2, conditions={0: 1, 1: 1})

    disjunctive = export_dynamic_qasm3(
        circuit.conditional("x", 2, any_of=({0: 1}, {1: 1}))
    )
    conjunctive = export_dynamic_qasm3(conjunction)

    assert "if ((c[0] == true) || (c[1] == true)) { x q[2]; }" in disjunctive
    # One clause keeps the exact text it had, so adding the disjunctive spelling
    # did not rewrite the conjunctive export.
    assert "if (c[0] == true && c[1] == true) { x q[2]; }" in conjunctive


def test_disjunctive_condition_requires_canonical_distinct_minimal_clauses() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=1)

    with pytest.raises(fqe.ValidationError, match="at least two clauses"):
        circuit.conditional("x", 1, any_of=({0: 1},))
    with pytest.raises(fqe.ValidationError, match="canonical, distinct, and minimal"):
        circuit.conditional("x", 1, any_of=({0: 1}, {0: 1}))
    # The second clause contains the first, so no trajectory reaches it alone.
    with pytest.raises(fqe.ValidationError, match="canonical, distinct, and minimal"):
        circuit.conditional("x", 1, any_of=({0: 1}, {0: 1, 1: 0}))
    with pytest.raises(fqe.ValidationError, match="binary values"):
        circuit.conditional("x", 1, any_of=({0: 1}, {1: 2}))
    with pytest.raises(fqe.ValidationError, match="at least one bit"):
        circuit.conditional("x", 1, any_of=({0: 1}, {}))
    with pytest.raises(fqe.ValidationError, match="any_of replaces"):
        circuit.conditional("x", 1, classical_bit=0, any_of=({0: 1}, {1: 1}))
    with pytest.raises(fqe.ValidationError, match="one mapping per clause"):
        circuit.conditional("x", 1, any_of=[(0, 1), (1, 1)])  # type: ignore[list-item]


def test_conditional_append_refuses_a_measure_or_reset_by_name() -> None:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)

    with pytest.raises(
        fqe.ValidationError, match="conditional reset is not constructible"
    ):
        circuit.conditional("reset", 1, classical_bit=0)
    with pytest.raises(
        fqe.ValidationError, match="conditional measure is not constructible"
    ):
        circuit.conditional("measure", 1, classical_bit=0)


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

    unsupported = CloudBackendProfile.simulator(2)
    rejected = fq.experimental.dynamic.assess_dynamic_backend(circuit, unsupported)
    assert not rejected.compatible
    assert rejected.required_classical_bits == 3
    assert "backend_does_not_declare_dynamic_circuit_support" in rejected.blockers

    supported = CloudBackendProfile(
        provider="test",
        name="dynamic-qpu",
        n_qubits=2,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=4,
    )
    accepted = fq.experimental.dynamic.assess_dynamic_backend(circuit, supported)
    assert accepted.compatible
    assert accepted.blockers == ()

    too_small = CloudBackendProfile(
        provider="test",
        name="small-register-qpu",
        n_qubits=2,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=2,
    )
    rejected = fq.experimental.dynamic.assess_dynamic_backend(circuit, too_small)
    assert rejected.blockers == ("required_classical_bits_exceed_backend_limit",)


def _dynamic_backend(**overrides) -> CloudBackendProfile:
    values = {
        "provider": "test",
        "name": "dynamic-qpu",
        "n_qubits": 4,
        "supports_openqasm": True,
        "supports_dynamic_circuits": True,
        "max_classical_bits": 4,
    }
    values.update(overrides)
    return CloudBackendProfile(**values)


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

    assert deployment.validate_deployment_package(package) is package
    assert package.qasm_version == 3.0
    assert package.shots == 256
    assert package.metadata["dynamic_circuit"] is True
    assert package.metadata["deployment_program_format"] == "openqasm-3"
    assert package.metadata["dynamic_backend_compatibility"]["compatible"] is True
    assert package.metadata["experiment"] == "bell-feedback"

    tampered = replace(package, qasm=package.qasm + "// tampered\n")
    with pytest.raises(
        deployment.DeploymentPackageIdentityError,
        match="QASM",
    ):
        deployment.validate_deployment_package(tampered)

    class RecordingProvider:
        def __init__(self):
            self.package = None

        def run(self, submitted):
            deployment.validate_deployment_package(submitted)
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
            backend=_dynamic_backend(n_qubits=1),
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
    coupling = CouplingMap.line(3)

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
        n_qubits=3,
        coupling_map=CouplingMap.line(3),
    )

    report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)
    package = create_dynamic_deployment_package(
        circuit,
        backend=backend,
        shots=32,
    )

    assert report.compatible
    assert deployment.validate_deployment_package(package) is package
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
    coupling = CouplingMap.line(3)
    routed = route_dynamic_circuit(circuit, coupling)
    backend = _dynamic_backend(n_qubits=3, coupling_map=coupling)

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


def test_dynamic_deployment_does_not_reuse_mismatched_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("cx", (0, 2), classical_bit=0)
    routed = route_dynamic_circuit(circuit, CouplingMap.line(3))
    backend = _dynamic_backend(n_qubits=3, coupling_map=CouplingMap.ring(3))
    route_calls = 0
    original_route = route_dynamic_circuit

    def count_routing(*args: object, **kwargs: object) -> DynamicCircuit:
        nonlocal route_calls
        route_calls += 1
        return original_route(*args, **kwargs)

    monkeypatch.setattr(
        "flagquantum.runtime.dynamic.deployment.route_dynamic_circuit",
        count_routing,
    )
    package = create_dynamic_deployment_package(routed, backend=backend, shots=32)

    assert route_calls == 1
    assert package.metadata["routing_reused"] is False
    assert package.metadata["routing_evidence"]["routing_reused"] is False
    assert package.metadata["routing_plan"]["coupling_edges"] == (
        backend.coupling_map.edges
    )


def test_dynamic_deployment_rejects_tampered_reused_routing() -> None:
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("cx", (0, 2), classical_bit=0)
    coupling = CouplingMap.line(3)
    routed = route_dynamic_circuit(circuit, coupling)
    routed_ir = routed.to_ir()
    metadata = dict(routed_ir.metadata)
    metadata["routing"] = dict(metadata["routing"])
    metadata["routing"]["inserted_swap_count"] = -1
    routed._ir_cache = replace(routed_ir, metadata=metadata)

    with pytest.raises(ValueError, match="non-negative integer"):
        create_dynamic_deployment_package(
            routed,
            backend=_dynamic_backend(n_qubits=3, coupling_map=coupling),
            shots=32,
        )
