import pytest
import torch

import flagquantum as fq


def test_local_dynamic_conformance_vectors_pass() -> None:
    result = fq.experimental.run_dynamic_conformance()
    assert result.passed
    assert result.cases == (
        ("active_reset", True),
        ("conditional_flip", True),
        ("qubit_reuse", True),
    )


def test_feature_assessment_is_provider_neutral_and_fail_closed() -> None:
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("h", 1, classical_bit=0, equals=0)

    local = fq.experimental.assess_dynamic_features(
        circuit, fq.experimental.LOCAL_TRAJECTORY_FEATURES
    )
    iqm = fq.experimental.assess_dynamic_features(
        circuit, fq.experimental.BRAKET_IQM_DYNAMIC_FEATURES
    )

    assert local.compatible
    assert not iqm.compatible
    assert "condition_value_unsupported" in iqm.blockers
    assert "conditional_gate_unsupported:h" in iqm.blockers
    assert (
        fq.experimental.BRAKET_IQM_DYNAMIC_FEATURES
        .returns_mid_circuit_measurements
        is False
    )


def test_feature_assessment_rejects_classical_read_before_measurement() -> None:
    circuit = fq.experimental.DynamicCircuit(1)
    circuit.conditional("x", 0, classical_bit=0)
    report = fq.experimental.assess_dynamic_features(
        circuit, fq.experimental.QISKIT_AER_DYNAMIC_FEATURES
    )
    assert report.blockers == ("classical_bit_read_before_measurement",)


def test_qiskit_aer_matches_deterministic_conformance_vectors() -> None:
    pytest.importorskip("qiskit_aer")
    result = fq.experimental.run_dynamic_conformance(
        fq.experimental.run_qiskit_aer_dynamic,
        implementation="qiskit_aer",
    )
    assert result.passed


def test_qiskit_aer_returns_final_and_mid_circuit_shots() -> None:
    pytest.importorskip("qiskit_aer")
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.x(0).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    result = fq.experimental.run_qiskit_aer_dynamic(circuit, shots=5, seed=7)

    assert result.execution_semantics == "qiskit_aer_dynamic_shots"
    assert torch.equal(result.samples, torch.ones((5, 2), dtype=torch.int64))
    assert torch.equal(result.classical_bits, torch.ones((5, 1), dtype=torch.int64))
