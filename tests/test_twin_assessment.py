"""Evidence-bound assessment scenarios for QPU digital-twin predictions."""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.twin import QPUDigitalTwin, TwinSupportEnvelope


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


def _support(twin, circuit, **overrides):
    values = {
        "snapshot_identity": twin.snapshot.identity,
        "physical_qubits": (3, 4),
        "supported_operations": ("h", "cx", "rx"),
        "maximum_instruction_count": 3,
        "verified_circuit_identities": (circuit.to_ir().content_hash,),
        "evidence_identity": "e" * 64,
        "verified_tv_error_radius": 0.04,
        "estimated_tv_error_radius": 0.08,
        "confidence_level": 0.95,
    }
    values.update(overrides)
    return TwinSupportEnvelope(**values)


def test_assessment_without_support_is_reference_only():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    assessment = twin.assess(circuit)

    assert assessment.decision == "physical_reference"
    assert assessment.actionable is False
    assert assessment.prediction is not None
    assert assessment.reasons == ("support_envelope_missing",)
    assert assessment.tv_error_radius is None


def test_exact_circuit_with_bound_is_verified():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    support = _support(twin, circuit)

    assessment = twin.assess(circuit, support=support)

    assert assessment.decision == "verified_prediction"
    assert assessment.actionable is True
    assert assessment.exact_circuit_verified is True
    assert assessment.structurally_supported is True
    assert assessment.tv_error_radius == pytest.approx(0.04)
    assert assessment.confidence_level == pytest.approx(0.95)
    assert assessment.support_envelope_identity == support.identity


def test_unseen_circuit_inside_envelope_is_bounded_estimate():
    twin = _twin()
    verified = fq.Circuit(2).h(0).cx(0, 1)
    unseen = fq.Circuit(2).rx(0, 0.2)

    assessment = twin.assess(unseen, support=_support(twin, verified))

    assert assessment.decision == "bounded_estimate"
    assert assessment.actionable is True
    assert assessment.exact_circuit_verified is False
    assert assessment.tv_error_radius == pytest.approx(0.08)
    assert assessment.reasons == ("exact_circuit_not_verified",)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"snapshot_identity": "a" * 64}, "snapshot_identity_mismatch"),
        ({"physical_qubits": (4, 3)}, "physical_mapping_mismatch"),
        ({"maximum_instruction_count": 1}, "outside_structural_support"),
        ({"supported_operations": ("h",)}, "outside_structural_support"),
    ],
)
def test_assessment_rejects_mismatched_or_unsupported_evidence(overrides, reason):
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    assessment = twin.assess(circuit, support=_support(twin, circuit, **overrides))

    assert assessment.decision == "unsupported"
    assert assessment.actionable is False
    assert reason in assessment.reasons
    assert assessment.tv_error_radius is None


def test_support_envelope_requires_confidence_for_error_bounds():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="requires confidence_level"):
        _support(twin, circuit, confidence_level=None)


def test_support_envelope_serialization_is_deterministic():
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    support = _support(twin, circuit)

    assert len(support.identity) == 64
    assert support.identity == _support(twin, circuit).identity
    assert support.to_dict()["physical_qubits"] == [3, 4]
