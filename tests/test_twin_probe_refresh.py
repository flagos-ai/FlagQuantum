"""Scenarios for deterministic, offline Twin probe-refresh planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from tests.test_twin_region_candidate_holdout import _holdout_circuits, _region
from tests.test_twin_region_release_assessment import _release

pytestmark = pytest.mark.integration


def _candidates():
    return (
        fq.Circuit(3).x(2).cx(1, 2),
        fq.Circuit(3).h(2).cx(0, 1),
        _holdout_circuits()[0],
        fq.Circuit(3).h(0).cx(0, 2),
    )


def _canonical(plan: fq.twin.TwinProbeRefreshPlan) -> bytes:
    return json.dumps(
        plan.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def test_developer_freezes_the_same_plan_independent_of_candidate_order() -> None:
    _, _, release = _release()
    candidates = _candidates()

    first = fq.twin.prepare_probe_refresh(
        release,
        candidates,
        maximum_tasks=4,
        shots=1024,
    )
    second = fq.twin.prepare_probe_refresh(
        release,
        tuple(reversed(candidates)),
        maximum_tasks=4,
        shots=1024,
    )

    assert _canonical(first) == _canonical(second)
    assert first.identity == second.identity
    assert first.planned_task_count == 4
    assert first.planned_shots == 4096
    assert len(first.selected_circuit_identities) == 2
    assert first.repetitions == 2
    assert "candidate_circuits_not_in_release" in first.coverage_gaps
    assert "task_budget_exhausted" in first.coverage_gaps


def test_structurally_unsupported_candidates_are_reported_not_substituted() -> None:
    _, _, release = _release()
    wrong_width = fq.Circuit(2).h(0).cx(0, 1)
    candidates = (*_candidates(), wrong_width)
    unsupported_identity = _candidates()[-1].to_ir().content_hash
    wrong_width_identity = wrong_width.to_ir().content_hash

    plan = fq.twin.prepare_probe_refresh(
        release,
        candidates,
        maximum_tasks=6,
        shots=512,
    )

    assert dict(plan.unsupported_candidates) == {
        unsupported_identity: ("physical_couplers_outside_region",),
        wrong_width_identity: ("physical_mapping_mismatch",),
    }
    assert unsupported_identity not in plan.selected_circuit_identities
    assert wrong_width_identity not in plan.selected_circuit_identities
    assert "unsupported_candidate_circuits" in plan.coverage_gaps
    assert plan.planned_task_count <= plan.maximum_tasks


def test_plan_materializes_only_selected_probes_through_existing_suite_path() -> None:
    _, candidate, release = _release()
    candidates = _candidates()
    plan = fq.twin.prepare_probe_refresh(
        release,
        candidates,
        maximum_tasks=4,
        shots=1024,
    )

    suite = plan.prepare_suite(candidate, candidates, name="refresh")

    assert isinstance(suite, fq.twin.TwinRegionValidationSuite)
    assert suite.planned_task_count == plan.planned_task_count
    assert suite.planned_shots == plan.planned_shots
    assert (
        tuple(
            experiment.prediction.circuit_identity for experiment in suite.experiments
        )
        == plan.selected_circuit_identities
    )
    assert all(experiment.shots == 1024 for experiment in suite.experiments)


def test_calibration_history_is_bound_and_observed_drift_affects_the_plan() -> None:
    _, candidate, release = _release()
    later = _region("2026-08-16 10:30:00", 0.97)
    history = fq.twin.build_calibration_history((candidate.twin, later.twin))

    plan = fq.twin.prepare_probe_refresh(
        release,
        _candidates(),
        calibration_history=history,
        maximum_tasks=4,
        shots=1024,
    )

    assert plan.calibration_history_identity is not None
    assert "calibration_changed_since_release" in plan.coverage_gaps
    assert "calibration_drift_observed" in plan.coverage_gaps
    assert "calibration_history_missing" not in plan.coverage_gaps


def test_validation_history_lowers_an_already_observed_circuit_priority() -> None:
    _, candidate, release = _release()
    later = _region("2026-08-16 10:30:00", 0.97)
    candidates = _candidates()[:2]
    observed_identity = candidates[0].to_ir().content_hash

    def series(region_twin, label: str):
        return fq.twin.TwinValidationSeries(
            snapshot_identity=region_twin.twin.snapshot.identity,
            circuit_identity=observed_identity,
            physical_qubits=(20, 27, 34),
            report_identities=tuple(
                hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
                for index in range(2)
            ),
            repetitions=2,
            total_shots=2048,
            mean_twin_hardware_total_variation=0.02,
            maximum_twin_hardware_total_variation=0.02,
            mean_ideal_hardware_total_variation=0.03,
            maximum_ideal_hardware_total_variation=0.03,
            mean_hardware_repeatability_total_variation=0.01,
            maximum_hardware_repeatability_total_variation=0.01,
            simultaneous_finite_shot_tv_radius=0.05,
            verified_tv_error_bound=0.07,
            confidence_level=0.95,
            supported_operations=("cx", "x"),
            maximum_instruction_count=2,
        )

    history = fq.twin.build_validation_history(
        (
            (candidate.twin, series(candidate, "candidate")),
            (later.twin, series(later, "later")),
        )
    )
    plan = fq.twin.prepare_probe_refresh(
        release,
        candidates,
        validation_history=history,
        maximum_tasks=2,
        shots=1024,
    )

    assert plan.selected_circuit_identities == (candidates[1].to_ir().content_hash,)
    assert plan.validation_history_identity is not None


def test_history_and_candidate_identity_mismatches_fail_closed() -> None:
    _, candidate, release = _release()
    later = _region("2026-08-16 10:30:00", 0.97)
    history = fq.twin.build_calibration_history((candidate.twin, later.twin))
    unrelated_release = replace(
        release,
        candidate_snapshot_identity="f" * 64,
    )

    with pytest.raises(ValueError, match="not anchored"):
        fq.twin.prepare_probe_refresh(
            unrelated_release,
            _candidates(),
            calibration_history=history,
            maximum_tasks=4,
            shots=1024,
        )

    duplicate = (_candidates()[0], _candidates()[0])
    with pytest.raises(ValueError, match="unique identities"):
        fq.twin.prepare_probe_refresh(
            release,
            duplicate,
            maximum_tasks=4,
            shots=1024,
        )


def test_budget_smaller_than_one_repeated_probe_returns_an_explained_empty_plan() -> (
    None
):
    _, _, release = _release()

    plan = fq.twin.prepare_probe_refresh(
        release,
        _candidates(),
        maximum_tasks=1,
        shots=1024,
    )

    assert plan.selected_circuit_identities == ()
    assert plan.planned_task_count == 0
    assert "task_budget_exhausted" in plan.coverage_gaps
    assert "insufficient_supported_circuits_for_suite" in plan.coverage_gaps
    with pytest.raises(ValueError, match="at least two selected probes"):
        plan.prepare_suite(_release()[1], _candidates(), name="empty")


def test_probe_refresh_plan_is_immutable_and_cannot_authorize_execution() -> None:
    _, _, release = _release()
    plan = fq.twin.prepare_probe_refresh(
        release,
        _candidates(),
        maximum_tasks=4,
        shots=1024,
    )

    with pytest.raises(FrozenInstanceError):
        plan.shots = 2048  # type: ignore[misc]
    assert not hasattr(plan, "submit")
    assert not hasattr(plan, "provider")
    assert not hasattr(plan, "routing_authorized")
    assert not hasattr(plan, "retry")
