"""Release-bound circuit assessment scenarios and identity tamper tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from flagquantum.twin import (
    TwinPrediction,
    TwinRegionModel,
    TwinRegionSupportAssessment,
    TwinReleaseAssessment,
)
from tests.test_twin_region_candidate_holdout import (
    _evaluate,
    _holdout_circuits,
    _reference_circuits,
    _region,
    _study_for,
    _with_predictions,
)

pytestmark = pytest.mark.integration

_REGION_MAPPING = (20, 27, 34)


def _release():
    incumbent = _region("2026-08-14 10:30:00", 0.96)
    candidate = _region("2026-08-15 10:30:00", 0.99)
    study = _with_predictions(_study_for(incumbent, candidate))
    evaluation = _evaluate(study)
    release = fq.twin.release_region_candidate(
        incumbent,
        candidate,
        study=study,
        evaluation=evaluation,
    )
    return incumbent, candidate, release


@pytest.fixture(scope="module")
def released():
    return _release()


def test_developer_asks_whether_an_exact_circuit_is_released(released) -> None:
    _, candidate, release = released
    circuit = _holdout_circuits()[0]

    assessment = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)

    assert assessment.status == "released_exact_circuit"
    assert assessment.reasons == ()
    assert assessment.release_identity == release.identity
    assert assessment.circuit_identity == circuit.to_ir().content_hash
    assert assessment.physical_qubits == _REGION_MAPPING
    assert isinstance(assessment.prediction, TwinPrediction)
    assert assessment.prediction.circuit_identity == circuit.to_ir().content_hash
    assert (
        assessment.prediction.snapshot_identity == release.candidate_snapshot_identity
    )
    assert assessment.prediction.n_wires == len(_REGION_MAPPING)


def test_release_claims_no_per_circuit_confidence_or_error_bound(released) -> None:
    _, candidate, release = released

    assessment = release.assess(
        candidate, _holdout_circuits()[1], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "released_exact_circuit"
    assert not hasattr(assessment, "confidence_level")
    assert not hasattr(assessment, "tv_error_bound")
    assert release.routing_authorized is False
    assert release.scope == "exact_circuits"


def test_loaded_release_covers_both_frozen_circuit_groups(released, tmp_path) -> None:
    _, candidate, release = released
    destination = tmp_path / "region-release.json"
    fq.twin.dump_region_release(release, destination)
    restored = fq.twin.load_region_release(destination)

    for circuit in (*_reference_circuits(), *_holdout_circuits()):
        assessment = restored.assess(
            candidate, circuit, physical_qubits=_REGION_MAPPING
        )
        assert assessment.status == "released_exact_circuit"
        assert assessment.release_identity == release.identity

    for circuit, identity in zip(
        _holdout_circuits(), release.holdout_circuit_identities, strict=True
    ):
        assert circuit.to_ir().content_hash == identity


def test_unknown_circuit_returns_reasons_instead_of_a_prediction(released) -> None:
    _, candidate, release = released
    circuit = fq.Circuit(3).h(2).cx(0, 1)

    assessment = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert assessment.reasons == ("circuit_identity_outside_release",)
    assert assessment.circuit_identity == circuit.to_ir().content_hash
    assert assessment.release_identity == release.identity


def test_support_assessment_distinguishes_an_unseen_covered_circuit(released) -> None:
    _, candidate, release = released
    circuit = fq.Circuit(3).h(2).cx(0, 1)

    assessment = release.assess_support(
        candidate, circuit, physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "within_envelope_unvalidated"
    assert assessment.prediction is None
    assert assessment.reasons == ("circuit_identity_not_validated",)
    assert assessment.circuit_identity == circuit.to_ir().content_hash
    assert assessment.release_identity == release.identity
    assert not hasattr(assessment, "confidence_level")
    assert not hasattr(assessment, "tv_error_bound")


def test_support_assessment_returns_a_prediction_only_for_an_exact_release(
    released,
) -> None:
    _, candidate, release = released
    circuit = _holdout_circuits()[0]

    support = release.assess_support(
        candidate, circuit, physical_qubits=_REGION_MAPPING
    )
    exact = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)

    assert support.status == "released_exact_circuit"
    assert support.prediction == exact.prediction
    assert support.reasons == ()


def test_support_assessment_refuses_structural_and_identity_mismatches(
    released,
) -> None:
    incumbent, candidate, release = released
    unsupported = fq.Circuit(3).h(0).cx(0, 2)

    structural = release.assess_support(
        candidate, unsupported, physical_qubits=_REGION_MAPPING
    )
    foreign = release.assess_support(
        incumbent, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert structural.status == "outside_envelope"
    assert structural.prediction is None
    assert structural.reasons == ("physical_couplers_outside_region",)
    assert foreign.status == "outside_envelope"
    assert foreign.prediction is None
    assert "candidate_region_identity_mismatch" in foreign.reasons
    assert "snapshot_identity_mismatch" in foreign.reasons


def test_support_assessment_refuses_a_tampered_frozen_envelope(released) -> None:
    _, candidate, release = released
    tampered = replace(
        release,
        maximum_circuit_depth=release.maximum_circuit_depth + 1,
    )

    assessment = tampered.assess_support(
        candidate, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "outside_envelope"
    assert assessment.prediction is None
    assert assessment.reasons == ("support_envelope_mismatch",)


def test_support_assessment_result_rejects_contradictory_states(released) -> None:
    _, candidate, release = released
    circuit = _holdout_circuits()[0]
    released_assessment = release.assess_support(
        candidate, circuit, physical_qubits=_REGION_MAPPING
    )
    prediction = released_assessment.prediction
    assert prediction is not None
    common = {
        "release_identity": release.identity,
        "circuit_identity": circuit.to_ir().content_hash,
        "physical_qubits": _REGION_MAPPING,
    }

    with pytest.raises(ValueError, match="requires deterministic reasons"):
        TwinRegionSupportAssessment(
            status="within_envelope_unvalidated",
            prediction=prediction,
            reasons=("circuit_identity_not_validated",),
            **common,
        )
    with pytest.raises(ValueError, match="requires only"):
        TwinRegionSupportAssessment(
            status="within_envelope_unvalidated",
            prediction=None,
            reasons=("target_mismatch",),
            **common,
        )
    with pytest.raises(ValueError, match="requires deterministic reasons"):
        TwinRegionSupportAssessment(
            status="outside_envelope",
            prediction=None,
            reasons=(),
            **common,
        )


def test_incumbent_model_cannot_borrow_a_candidate_release(released) -> None:
    incumbent, _, release = released

    assessment = release.assess(
        incumbent, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert "candidate_region_identity_mismatch" in assessment.reasons
    assert "snapshot_identity_mismatch" in assessment.reasons


def test_recaptured_model_with_the_same_mapping_fails_closed(released) -> None:
    _, candidate, release = released
    recaptured = replace(candidate, twin=_region("2026-08-15 10:30:00", 0.97).twin)
    assert recaptured.region == candidate.region

    assessment = release.assess(
        recaptured, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert "snapshot_identity_mismatch" in assessment.reasons
    assert "candidate_region_identity_mismatch" in assessment.reasons


def test_release_snapshot_tamper_fails_closed(released) -> None:
    _, candidate, release = released
    tampered = replace(release, candidate_snapshot_identity="0" * 64)

    assessment = tampered.assess(
        candidate, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert assessment.reasons == ("snapshot_identity_mismatch",)
    assert assessment.release_identity == tampered.identity


def test_target_tamper_fails_closed(released) -> None:
    _, candidate, release = released
    tampered = replace(release, backend_name="Shenglian-2")

    assessment = tampered.assess(
        candidate, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert assessment.reasons == ("target_mismatch",)
    assert tampered.target == "quafu:Shenglian-2"


@pytest.mark.parametrize(
    "mapping",
    (
        (27, 20, 34),
        (20, 27, 34, 40),
        (20, 27),
    ),
)
def test_wrong_ordered_mapping_fails_closed(released, mapping) -> None:
    _, candidate, release = released
    circuit = (
        _holdout_circuits()[0] if len(mapping) == 3 else fq.Circuit(2).h(0).cx(0, 1)
    )

    assessment = release.assess(candidate, circuit, physical_qubits=mapping)

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert "physical_mapping_mismatch" in assessment.reasons
    assert assessment.physical_qubits == tuple(mapping)


def test_structurally_unsupported_circuit_fails_closed(released) -> None:
    _, candidate, release = released
    circuit = fq.Circuit(3).h(0).cx(0, 2)

    assessment = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)

    assert assessment.status == "outside_release"
    assert assessment.prediction is None
    assert "physical_couplers_outside_region" in assessment.reasons
    assert "circuit_identity_outside_release" in assessment.reasons


def test_assessment_is_deterministic_and_never_mutates_its_inputs(released) -> None:
    _, candidate, release = released
    circuit = _reference_circuits()[0]
    before = (release.identity, candidate.identity, candidate.twin.snapshot.identity)

    first = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)
    second = release.assess(candidate, circuit, physical_qubits=_REGION_MAPPING)

    assert first == second
    assert first.status == "released_exact_circuit"
    assert (release.identity, candidate.identity, candidate.twin.snapshot.identity) == (
        before
    )
    with pytest.raises(FrozenInstanceError):
        first.status = "outside_release"  # type: ignore[misc]


def test_assessment_rejects_a_non_regional_model(released) -> None:
    _, candidate, release = released

    with pytest.raises(TypeError, match="region_twin must be a TwinRegionModel"):
        release.assess(
            candidate.twin, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
        )


def test_assessment_result_rejects_contradictory_state(released) -> None:
    _, candidate, release = released
    released_assessment = release.assess(
        candidate, _holdout_circuits()[0], physical_qubits=_REGION_MAPPING
    )
    prediction = released_assessment.prediction
    assert prediction is not None

    common = {
        "release_identity": release.identity,
        "circuit_identity": prediction.circuit_identity,
        "physical_qubits": _REGION_MAPPING,
    }
    with pytest.raises(ValueError, match="requires one prediction and no reasons"):
        TwinReleaseAssessment(
            status="released_exact_circuit",
            prediction=None,
            reasons=(),
            **common,
        )
    with pytest.raises(ValueError, match="requires one prediction and no reasons"):
        TwinReleaseAssessment(
            status="released_exact_circuit",
            prediction=prediction,
            reasons=("target_mismatch",),
            **common,
        )
    with pytest.raises(ValueError, match="deterministic reasons and no prediction"):
        TwinReleaseAssessment(
            status="outside_release",
            prediction=None,
            reasons=(),
            **common,
        )
    other = release.assess(
        candidate, _holdout_circuits()[1], physical_qubits=_REGION_MAPPING
    )
    with pytest.raises(ValueError, match="does not match the assessed circuit"):
        TwinReleaseAssessment(
            status="released_exact_circuit",
            prediction=other.prediction,
            reasons=(),
            **common,
        )


def test_regional_model_rejects_a_foreign_mapping_order(released) -> None:
    _, candidate, release = released
    assert isinstance(candidate, TwinRegionModel)

    with pytest.raises(
        ValueError, match="exactly match the composed region wire order"
    ):
        candidate.predict(_holdout_circuits()[0], physical_qubits=(27, 20, 34))
