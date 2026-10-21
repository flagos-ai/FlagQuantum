import pytest
import torch

import flagquantum as fq
import flagquantum.ecosystem.qiskit as fqq
import flagquantum.runtime.dynamic.conformance as fqdc
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.integration


def test_local_dynamic_conformance_vectors_pass() -> None:
    result = fqdc.run_dynamic_conformance()
    assert result.passed
    assert result.cases == (
        ("active_reset", True),
        ("conditional_flip", True),
        ("qubit_reuse", True),
    )


def test_feature_assessment_is_provider_neutral_and_fail_closed() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("h", 1, classical_bit=0, equals=0)

    local = fqdc.assess_dynamic_features(circuit, fqdc.LOCAL_TRAJECTORY_FEATURES)
    iqm = fqdc.assess_dynamic_features(circuit, fqdc.BRAKET_IQM_DYNAMIC_FEATURES)

    assert local.compatible
    assert not iqm.compatible
    assert "condition_value_unsupported" in iqm.blockers
    assert "conditional_gate_unsupported:h" in iqm.blockers
    assert fqdc.BRAKET_IQM_DYNAMIC_FEATURES.returns_mid_circuit_measurements is False


def test_declared_disjunctive_feature_matches_what_each_path_accepts() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=1)
    circuit.conditional("x", 1, any_of=({0: 1}, {1: 1}))

    # The local trajectory path declares the feature, so assessment has to agree
    # with the constructor that can now express it; the one-bit provider paths
    # declare its absence, so they have to report the blocker rather than the
    # narrower condition-width one.
    local = fqdc.assess_dynamic_features(circuit, fqdc.LOCAL_TRAJECTORY_FEATURES)
    aer = fqdc.assess_dynamic_features(circuit, fqdc.QISKIT_AER_DYNAMIC_FEATURES)
    iqm = fqdc.assess_dynamic_features(circuit, fqdc.BRAKET_IQM_DYNAMIC_FEATURES)

    assert local.compatible
    assert fqdc.LOCAL_TRAJECTORY_FEATURES.supports_disjunctive_conditions is True
    assert "disjunctive_conditions_unsupported" in aer.blockers
    assert "disjunctive_conditions_unsupported" in iqm.blockers


def test_feature_assessment_rejects_classical_read_before_measurement() -> None:
    circuit = DynamicCircuit.from_ir(
        CircuitIR(
            1,
            (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),),
        )
    )
    report = fqdc.assess_dynamic_features(circuit, fqdc.QISKIT_AER_DYNAMIC_FEATURES)
    assert report.blockers == ("classical_bit_read_before_measurement",)


@pytest.mark.qiskit
def test_qiskit_aer_matches_deterministic_conformance_vectors() -> None:
    pytest.importorskip("qiskit_aer")
    result = fqdc.run_dynamic_conformance(
        fqq.run_qiskit_aer_dynamic,
        implementation="qiskit_aer",
    )
    assert result.passed


@pytest.mark.qiskit
def test_qiskit_aer_returns_final_and_mid_circuit_shots() -> None:
    pytest.importorskip("qiskit_aer")
    circuit = DynamicCircuit(2)
    circuit.x(0).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    result = fqq.run_qiskit_aer_dynamic(circuit, shots=5, seed=7)

    assert result.execution_semantics == "qiskit_aer_dynamic_shots"
    assert torch.equal(result.samples, torch.ones((5, 2), dtype=torch.int64))
    assert torch.equal(result.classical_bits, torch.ones((5, 1), dtype=torch.int64))


@pytest.mark.qiskit
def test_qiskit_qasm3_round_trip_preserves_dynamic_feedback() -> None:
    pytest.importorskip("qiskit_aer")
    circuit = DynamicCircuit(2)
    circuit.x(0).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    result = fqq.run_qiskit_aer_qasm3_round_trip(circuit, shots=8, seed=9)

    assert torch.equal(result.final_samples, torch.ones((8, 2), dtype=torch.int64))
    assert torch.equal(
        result.mid_circuit_measurements,
        torch.ones((8, 1), dtype=torch.int64),
    )
    assert result.provider_metadata["transport"] == "openqasm3"


@pytest.mark.qiskit
def test_random_branch_statistics_match_local_aer_and_qasm_round_trip() -> None:
    pytest.importorskip("qiskit_aer")
    circuit = DynamicCircuit(2)
    circuit.h(0).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    local = fq.experimental.dynamic.run_dynamic(circuit, shots=4096, seed=41)
    aer = fqq.run_qiskit_aer_dynamic(circuit, shots=4096, seed=41)
    qasm = fqq.run_qiskit_aer_qasm3_round_trip(circuit, shots=4096, seed=41)
    probabilities = [
        float(result.final_samples[:, 0].float().mean())
        for result in (local, aer, qasm)
    ]

    assert all(abs(probability - 0.5) < 0.04 for probability in probabilities)
    assert max(probabilities) - min(probabilities) < 0.05
    assert torch.equal(
        fq.experimental.dynamic.run_dynamic(circuit, shots=32, seed=7).final_samples,
        fq.experimental.dynamic.run_dynamic(circuit, shots=32, seed=7).final_samples,
    )


def test_dynamic_result_projects_to_canonical_execution_result() -> None:
    circuit = DynamicCircuit(1).x(0).measure(0, classical_bit=0)
    dynamic = fq.experimental.dynamic.run_dynamic(circuit, shots=4, seed=3)
    result = dynamic.to_execution_result()

    assert result.samples is dynamic.final_samples
    assert tuple(item.kind for item in result.measurements) == (
        "sample",
        "mid_circuit_measurement",
    )
    assert result.runtime["mid_circuit_measurements_available"] is True
