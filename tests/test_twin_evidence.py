"""Empirical evidence-report scenarios for QPU digital-twin predictions."""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.twin import QPUDigitalTwin, TwinEvidenceEnvelope


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q3": {"T1": 40.0, "T2": 60.0, "fidelity": 1.0, "length": 6.4e-8},
            "Q4": {"T1": 35.0, "T2": 50.0, "fidelity": 1.0, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [3, 4],
                "fidelity": 1.0,
                "length": 2.24e-7,
            }
        },
    }


def _twin():
    return QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(), backend_name="Baihua", physical_qubits=(3, 4)
    )


def _evidence(twin, circuit, **overrides):
    values = {
        "snapshot_identity": twin.snapshot.identity,
        "physical_qubits": (3, 4),
        "supported_operations": ("h", "cx", "rx"),
        "maximum_instruction_count": 3,
        "verified_circuit_identities": (circuit.to_ir().content_hash,),
        "evidence_identity": "e" * 64,
        "verified_tv_error_bound": 0.04,
        "estimated_tv_error_bound": 0.08,
        "confidence_level": 0.95,
    }
    values.update(overrides)
    return TwinEvidenceEnvelope(**values)


def test_report_without_evidence_is_unverified():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    report = twin.evidence_report(circuit)

    assert report.status == "unverified"
    assert report.prediction is not None
    assert report.reasons == ("evidence_envelope_missing",)
    assert report.tv_error_bound is None
    assert not hasattr(report, "actionable")


def test_exact_circuit_with_bound_is_verified():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = _evidence(twin, circuit)

    report = twin.evidence_report(circuit, evidence=evidence)

    assert report.status == "exact_circuit_verified"
    assert report.exact_circuit_verified is True
    assert report.structurally_supported is True
    assert report.tv_error_bound == pytest.approx(0.04)
    assert report.confidence_level == pytest.approx(0.95)
    assert report.evidence_envelope_identity == evidence.identity


def test_unseen_circuit_inside_envelope_reports_structural_evidence():
    twin = _twin()
    verified = fq.Circuit(2).h(0).cx(0, 1)
    unseen = fq.Circuit(2).rx(0, 0.2)

    report = twin.evidence_report(unseen, evidence=_evidence(twin, verified))

    assert report.status == "within_evidence_envelope"
    assert report.exact_circuit_verified is False
    assert report.tv_error_bound == pytest.approx(0.08)
    assert report.reasons == ("exact_circuit_not_verified",)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"snapshot_identity": "a" * 64}, "snapshot_identity_mismatch"),
        ({"physical_qubits": (4, 3)}, "physical_mapping_mismatch"),
        ({"maximum_instruction_count": 1}, "outside_structural_support"),
        ({"supported_operations": ("h",)}, "outside_structural_support"),
    ],
)
def test_report_marks_mismatched_or_outside_evidence_out_of_scope(overrides, reason):
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    report = twin.evidence_report(
        circuit, evidence=_evidence(twin, circuit, **overrides)
    )

    assert report.status == "out_of_scope"
    assert reason in report.reasons
    assert report.tv_error_bound is None


def test_evidence_envelope_requires_confidence_for_error_bounds():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="requires confidence_level"):
        _evidence(twin, circuit, confidence_level=None)


def test_evidence_envelope_serialization_is_deterministic():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = _evidence(twin, circuit)

    assert len(evidence.identity) == 64
    assert evidence.identity == _evidence(twin, circuit).identity
    assert evidence.to_dict()["physical_qubits"] == [3, 4]
