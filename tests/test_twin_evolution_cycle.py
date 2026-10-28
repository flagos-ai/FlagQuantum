"""Scenarios for immutable, human-gated Twin evolution-cycle records."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from flagquantum.twin import TwinPrediction
from tests.test_twin_candidate_fit import _estimate, _incumbent
from tests.test_twin_candidate_suite import _result, _submission

pytestmark = pytest.mark.integration


def _freshness(incumbent, status="refresh_due"):
    return fq.twin.TwinFreshnessAssessment(
        status=status,
        snapshot_identity=incumbent.snapshot.identity,
        evaluated_at="2026-08-15T09:00:00+08:00",
        calibration_age_seconds=81_000.0,
        validation_snapshot_age_seconds=None,
        next_refresh_at="2026-08-15T08:30:00+08:00",
        calibration_drift=None,
        validation_series=None,
        reasons=() if status == "current_under_policy" else ("refresh_due",),
    )


def _plan(incumbent):
    return fq.twin.TwinProbeRefreshPlan(
        release_identity="1" * 64,
        candidate_region_identity="2" * 64,
        candidate_snapshot_identity=incumbent.snapshot.identity,
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        candidate_circuit_identities=("3" * 64, "4" * 64),
        selected_circuit_identities=("3" * 64, "4" * 64),
        unsupported_candidates=(),
        covered_directed_couplers=((20, 27),),
        coverage_gaps=(),
        calibration_history_identity=None,
        validation_history_identity=None,
        shots=1024,
        maximum_tasks=4,
    )


def _prediction(base: TwinPrediction, probabilities: tuple[float, ...]):
    distance = 0.5 * sum(
        abs(ideal - predicted)
        for ideal, predicted in zip(
            base.ideal_probabilities, probabilities, strict=True
        )
    )
    return replace(
        base,
        twin_probabilities=probabilities,
        total_variation_from_ideal=distance,
    )


def _holdout_evaluation(incumbent, candidate):
    circuits = (
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).x(1).cx(1, 0),
        fq.Circuit(2).h(1).cx(1, 0).x(0),
    )
    base = fq.twin.prepare_candidate_suite(
        incumbent,
        candidate,
        circuits,
        name="evolution-holdout",
        shots=1024,
    )
    suite = fq.twin.TwinCandidateSuite(
        tuple(
            replace(
                trial,
                incumbent_prediction=_prediction(
                    trial.incumbent_prediction, (0.70, 0.0, 0.0, 0.30)
                ),
                experiment=replace(
                    trial.experiment,
                    prediction=_prediction(
                        trial.experiment.prediction, (0.98, 0.0, 0.0, 0.02)
                    ),
                ),
            )
            for trial in base.trials
        )
    )
    submissions = tuple(
        _submission(trial, index) for index, trial in enumerate(suite.trials, start=1)
    )
    results = tuple(_result(submission) for submission in submissions)
    evaluation = suite.validate_results(
        submissions,
        results,
        circuits=circuits,
        confidence_level=0.95,
    )
    return evaluation


def _complete_cycle():
    incumbent = _incumbent()
    freshness = _freshness(incumbent)
    plan = _plan(incumbent)
    estimate = _estimate(incumbent)
    candidate = fq.twin.fit_candidate(
        incumbent,
        estimate,
        captured_at="2026-08-15T10:30:00+08:00",
    )
    evaluation = _holdout_evaluation(incumbent, candidate)
    prepared = fq.twin.prepare_evolution_cycle(
        incumbent=incumbent,
        freshness=freshness,
        probe_plan=plan,
    )
    estimated = prepared.with_parameter_estimate(estimate)
    fitted = estimated.with_candidate(candidate)
    completed = fitted.with_holdout_evaluation(evaluation)
    return prepared, estimated, fitted, completed, plan, estimate, candidate, evaluation


def test_cycle_records_immutable_transitions_and_recommends_release() -> None:
    prepared, estimated, fitted, completed, *_ = _complete_cycle()

    assert prepared.stage == "prepared"
    assert prepared.recommended_action == "collect_more_evidence"
    assert prepared.parameter_estimate_identity is None
    assert estimated.stage == "parameter_estimated"
    assert fitted.stage == "candidate_fitted"
    assert completed.stage == "holdout_evaluated"
    assert completed.holdout_decision == "improved"
    assert completed.recommended_action == "release_candidate"
    assert (
        prepared.identity != estimated.identity != fitted.identity != completed.identity
    )
    with pytest.raises(FrozenInstanceError):
        completed.stage = "prepared"  # type: ignore[misc]


def test_cycle_fails_closed_on_skipped_or_repeated_transitions() -> None:
    prepared, estimated, fitted, completed, _, estimate, candidate, evaluation = (
        _complete_cycle()
    )

    with pytest.raises(ValueError, match="candidate can only follow"):
        prepared.with_candidate(candidate)
    with pytest.raises(ValueError, match="prepared stage"):
        estimated.with_parameter_estimate(estimate)
    with pytest.raises(ValueError, match="candidate fitting"):
        completed.with_holdout_evaluation(evaluation)
    with pytest.raises(ValueError, match="holdout evaluation can only follow"):
        estimated.with_holdout_evaluation(evaluation)
    assert fitted.recommended_action == "collect_more_evidence"


def test_cycle_round_trip_resumes_without_provider_operations(tmp_path) -> None:
    _, estimated, _, _, _, _, candidate, evaluation = _complete_cycle()
    destination = tmp_path / "evolution-cycle.json"

    fq.twin.dump_evolution_cycle(estimated, destination)
    fq.twin.dump_evolution_cycle(estimated, destination)
    restored = fq.twin.load_evolution_cycle(destination)
    resumed = restored.with_candidate(candidate).with_holdout_evaluation(evaluation)

    assert restored == estimated
    assert resumed.recommended_action == "release_candidate"
    assert destination.stat().st_mode & 0o777 == 0o600
    for name in (
        "submit",
        "poll",
        "retry",
        "promote",
        "release",
        "route",
        "notify",
    ):
        assert not hasattr(restored, name)


def test_cycle_references_are_smaller_than_their_source_artifacts() -> None:
    _, _, _, completed, plan, estimate, candidate, evaluation = _complete_cycle()
    record_size = len(json.dumps(completed.to_dict(), sort_keys=True))
    source_size = sum(
        len(json.dumps(payload, sort_keys=True))
        for payload in (
            plan.to_dict(),
            estimate.to_dict(),
            candidate.noise_model.to_dict(),
            evaluation.to_dict(),
        )
    )

    assert record_size < source_size
    assert "evaluations" not in completed.to_dict()
    assert "noise_model" not in completed.to_dict()


def test_cycle_rejects_changed_bindings_and_tampered_persistence(tmp_path) -> None:
    incumbent = _incumbent()
    plan = _plan(incumbent)
    with pytest.raises(ValueError, match="freshness"):
        fq.twin.prepare_evolution_cycle(
            incumbent=incumbent,
            freshness=replace(_freshness(incumbent), snapshot_identity="a" * 64),
            probe_plan=plan,
        )
    with pytest.raises(ValueError, match="probe plan"):
        fq.twin.prepare_evolution_cycle(
            incumbent=incumbent,
            freshness=_freshness(incumbent),
            probe_plan=replace(plan, target="quafu:Baihua"),
        )

    completed = _complete_cycle()[3]
    destination = tmp_path / "tampered-cycle.json"
    fq.twin.dump_evolution_cycle(completed, destination)
    payload = json.loads(destination.read_text())
    payload["recommended_action"] = "retain_incumbent"
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid Twin evolution cycle"):
        fq.twin.load_evolution_cycle(destination)


def test_current_freshness_recommends_retaining_the_incumbent() -> None:
    incumbent = _incumbent()
    cycle = fq.twin.prepare_evolution_cycle(
        incumbent=incumbent,
        freshness=_freshness(incumbent, "current_under_policy"),
        probe_plan=_plan(incumbent),
    )

    assert cycle.recommended_action == "retain_incumbent"
