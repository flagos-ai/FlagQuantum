from dataclasses import replace
from math import pi

import pytest

import flagquantum as fq


def _iqm_backend(**overrides) -> fq.CloudBackendProfile:
    values = {
        "provider": "amazon-braket",
        "name": "iqm-test",
        "n_wires": 4,
        "supports_openqasm": True,
        "supports_dynamic_circuits": True,
        "max_classical_bits": 8,
        "dynamic_dialect": "braket_iqm",
        "metadata": {"dynamic_qubit_groups": ((0, 1), (2, 3))},
    }
    values.update(overrides)
    return fq.CloudBackendProfile(**values)


def test_braket_iqm_export_lowers_feedback_and_active_reset() -> None:
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.x(0)
    circuit.measure(0, classical_bit=3)
    circuit.conditional("x", 1, classical_bit=3)
    circuit.reset(1)

    qasm = fq.experimental.export_dynamic_qasm3_for_backend(
        circuit, _iqm_backend(n_wires=2)
    )

    assert "#pragma braket verbatim" in qasm
    assert f"prx({pi!r}, 0.0) $0;" in qasm
    assert "measure_ff(0) $0;" in qasm
    assert f"cc_prx({pi!r}, 0.0, 0) $1;" in qasm
    assert "measure_ff(1) $1;" in qasm
    assert f"cc_prx({pi!r}, 0.0, 1) $1;" in qasm
    assert "b[0] = measure $0;" in qasm
    assert "b[1] = measure $1;" in qasm
    assert "qubit[" not in qasm


def test_braket_iqm_reuses_latest_unique_feedback_key() -> None:
    circuit = fq.experimental.DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 2, classical_bit=0)
    circuit.measure(1, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.25})

    qasm = fq.experimental.export_dynamic_qasm3_for_backend(
        circuit,
        _iqm_backend(n_wires=3, metadata={"dynamic_qubit_groups": ((0, 1, 2),)}),
    )

    assert "measure_ff(0) $0;" in qasm
    assert f"cc_prx({pi!r}, 0.0, 0) $2;" in qasm
    assert "measure_ff(1) $1;" in qasm
    assert "cc_prx(0.25, 0.0, 1) $1;" in qasm


@pytest.mark.parametrize(
    ("build", "blocker"),
    (
        (
            lambda: fq.experimental.DynamicCircuit(2).conditional(
                "x", 1, classical_bit=0
            ),
            "feedback_bit_was_read_before_measurement",
        ),
        (
            lambda: fq.experimental.DynamicCircuit(2)
            .measure(0, classical_bit=0)
            .conditional("x", 1, classical_bit=0, equals=0),
            "conditions_require_one_bit_equal_to_one",
        ),
        (
            lambda: fq.experimental.DynamicCircuit(2)
            .measure(0, classical_bit=0)
            .conditional("h", 1, classical_bit=0),
            "conditional_gate_cannot_lower_to_cc_prx",
        ),
    ),
)
def test_braket_iqm_assessment_rejects_unsupported_feedback(build, blocker) -> None:
    report = fq.experimental.assess_dynamic_backend(build(), _iqm_backend())
    assert not report.compatible
    assert any(blocker in item for item in report.blockers)


def test_braket_iqm_rejects_cross_group_and_multiple_controllers() -> None:
    cross_group = fq.experimental.DynamicCircuit(3)
    cross_group.measure(0, classical_bit=0)
    cross_group.conditional("x", 2, classical_bit=0)
    report = fq.experimental.assess_dynamic_backend(cross_group, _iqm_backend())
    assert "braket_iqm_feedback_pair_outside_dynamic_qubit_group" in report.blockers

    multiple = fq.experimental.DynamicCircuit(3)
    multiple.measure(0, classical_bit=0)
    multiple.conditional("x", 2, classical_bit=0)
    multiple.measure(1, classical_bit=1)
    multiple.conditional("x", 2, classical_bit=1)
    backend = _iqm_backend(
        metadata={"dynamic_qubit_groups": ((0, 1, 2),)}
    )
    report = fq.experimental.assess_dynamic_backend(multiple, backend)
    assert "braket_iqm_target_has_multiple_feedback_controllers" in report.blockers


def test_braket_iqm_rejects_mid_circuit_measurement_without_feed_forward() -> None:
    circuit = fq.experimental.DynamicCircuit(1).measure(0, classical_bit=0)
    report = fq.experimental.assess_dynamic_backend(
        circuit,
        _iqm_backend(
            n_wires=1,
            metadata={"dynamic_qubit_groups": ((0,),)},
        ),
    )
    assert "braket_iqm_mid_circuit_measurement_requires_feed_forward" in report.blockers


def test_braket_iqm_deployment_is_dialect_sealed() -> None:
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    package = fq.experimental.create_dynamic_deployment_package(
        circuit, backend=_iqm_backend(n_wires=2), shots=25
    )

    assert fq.deployment.validate_deployment_package(package) is package
    assert package.metadata["dynamic_dialect"] == "braket_iqm"
    assert package.metadata["mid_circuit_measurements_returned"] is False
    assert package.metadata["dynamic_backend_compatibility"]["dynamic_dialect"] == (
        "braket_iqm"
    )

    generic_backend = replace(package.backend, dynamic_dialect=None)
    tampered = replace(package, backend=generic_backend)
    with pytest.raises(fq.deployment.DeploymentPackageIdentityError, match="QASM"):
        fq.deployment.validate_deployment_package(tampered)
