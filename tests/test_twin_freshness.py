"""Caller-policy freshness scenarios for frozen QPU Twins."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration

_CIRCUIT_IDENTITY = fq.Circuit(2).h(0).cx(0, 1).to_ir().content_hash


def _twin(
    *,
    captured_at: str,
    changed: bool = False,
    backend: str = "Shenglian",
):
    chip_info = {
        "calibration_time": captured_at,
        "basis_gates": ["h", "cx"],
        "qubits_info": {
            "Q20": {
                "T1": 44.0 if changed else 40.0,
                "T2": 54.0 if changed else 60.0,
                "fidelity": 0.98 if changed else 0.99,
                "length": 7.04e-8 if changed else 6.4e-8,
            },
            "Q27": {
                "T1": 35.0,
                "T2": 50.0,
                "fidelity": 0.99,
                "length": 6.4e-8,
            },
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 0.97 if changed else 0.98,
                "length": 2.464e-7 if changed else 2.24e-7,
            }
        },
    }
    readout = (
        (
            ((0.95, 0.05), (0.04, 0.96)),
            ((0.97, 0.03), (0.02, 0.98)),
        )
        if changed
        else (
            ((0.98, 0.02), (0.03, 0.97)),
            ((0.97, 0.03), (0.02, 0.98)),
        )
    )
    return fq.twin.from_quafu_chip_info(
        chip_info,
        target=f"quafu:{backend}",
        qubits=(20, 27),
        readout_confusion_matrices=readout,
    )


def _series(twin, *, label: str, distance: float = 0.06):
    reports = tuple(
        hashlib.sha256(f"{label}-{index}".encode()).hexdigest() for index in range(2)
    )
    return fq.twin.TwinValidationSeries(
        snapshot_identity=twin.snapshot.identity,
        circuit_identity=_CIRCUIT_IDENTITY,
        physical_qubits=(20, 27),
        report_identities=reports,
        repetitions=2,
        total_shots=2048,
        mean_twin_hardware_total_variation=distance,
        maximum_twin_hardware_total_variation=distance,
        mean_ideal_hardware_total_variation=0.12,
        maximum_ideal_hardware_total_variation=0.12,
        mean_hardware_repeatability_total_variation=0.03,
        maximum_hardware_repeatability_total_variation=0.03,
        simultaneous_finite_shot_tv_radius=0.05,
        verified_tv_error_bound=distance + 0.05,
        confidence_level=0.95,
        supported_operations=("cx", "h", "measure"),
        maximum_instruction_count=2,
    )


def _history_fixture():
    first = _twin(captured_at="2026-08-14T10:30:00+08:00")
    current = _twin(
        captured_at="2026-08-14T11:30:00+08:00",
        changed=True,
    )
    calibration = fq.twin.build_calibration_history((first, current))
    validation = fq.twin.build_validation_history(
        (
            (first, _series(first, label="state-01", distance=0.08)),
            (current, _series(current, label="state-02", distance=0.06)),
        )
    )
    return first, current, calibration, validation


def _policy(**changes):
    values = {
        "maximum_calibration_age": timedelta(hours=2),
        "maximum_validation_age": timedelta(hours=3),
        "maximum_relative_t1_drift": 0.20,
        "maximum_relative_t2_drift": 0.20,
        "maximum_readout_tv_drift": 0.05,
        "maximum_relative_gate_duration_drift": 0.20,
        "maximum_verified_tv_error_bound": 0.20,
    }
    values.update(changes)
    return fq.twin.TwinFreshnessPolicy(**values)


def test_matching_evidence_is_current_under_explicit_policy():
    _, current, calibration, validation = _history_fixture()

    assessment = _policy().assess(
        current,
        calibration_history=calibration,
        validation_history=validation,
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )

    assert assessment.status == "current_under_policy"
    assert assessment.reasons == ()
    assert assessment.snapshot_identity == current.snapshot.identity
    assert assessment.calibration_age_seconds == 1800
    assert assessment.validation_snapshot_age_seconds == 1800
    assert assessment.next_refresh_at == "2026-08-14T13:30:00+08:00"
    assert assessment.calibration_drift is calibration.latest_interval_drift
    assert assessment.validation_series is validation.validation_series[-1]


def test_age_drift_and_bound_thresholds_report_refresh_reasons():
    _, current, calibration, validation = _history_fixture()
    policy = _policy(
        maximum_calibration_age=timedelta(minutes=15),
        maximum_validation_age=timedelta(minutes=15),
        maximum_relative_t1_drift=0.05,
        maximum_relative_t2_drift=0.05,
        maximum_readout_tv_drift=0.01,
        maximum_relative_gate_duration_drift=0.05,
        maximum_verified_tv_error_bound=0.10,
    )
    as_of = datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc)

    assessment = policy.assess(
        current,
        calibration_history=calibration,
        validation_history=validation,
        as_of=as_of,
    )

    assert assessment.status == "refresh_due"
    assert set(assessment.reasons) == {
        "maximum_calibration_age_exceeded",
        "maximum_relative_t1_drift_exceeded",
        "maximum_relative_t2_drift_exceeded",
        "maximum_readout_tv_drift_exceeded",
        "maximum_relative_gate_duration_drift_exceeded",
        "maximum_validation_age_exceeded",
        "maximum_verified_tv_error_bound_exceeded",
    }
    assert assessment.next_refresh_at == as_of.isoformat()


def test_validation_for_older_snapshot_is_due_not_borrowed():
    first, current, calibration, _ = _history_fixture()
    older = _twin(captured_at="2026-08-14T09:30:00+08:00")
    stale_validation = fq.twin.build_validation_history(
        (
            (older, _series(older, label="state-00", distance=0.08)),
            (first, _series(first, label="state-01", distance=0.07)),
        )
    )

    assessment = _policy().assess(
        current,
        calibration_history=calibration,
        validation_history=stale_validation,
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )

    assert assessment.status == "refresh_due"
    assert "current_snapshot_not_validated" in assessment.reasons
    assert assessment.validation_series is stale_validation.validation_series[-1]


def test_required_missing_histories_are_insufficient_evidence():
    _, current, _, _ = _history_fixture()

    assessment = _policy().assess(
        current,
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )

    assert assessment.status == "insufficient_evidence"
    assert assessment.reasons == (
        "calibration_history_required",
        "validation_history_required",
    )
    assert assessment.next_refresh_at is None


def test_unavailable_drift_component_is_insufficient_not_zero():
    reference = _twin(captured_at="2026-08-14T10:30:00+08:00")
    current = _twin(captured_at="2026-08-14T11:30:00+08:00")
    without_readout = tuple(
        fq.twin.from_quafu_chip_info(
            {
                "calibration_time": twin.snapshot.captured_at,
                "basis_gates": ["h", "cx"],
                "qubits_info": {
                    "Q20": {
                        "T1": 40.0,
                        "T2": 60.0,
                        "fidelity": 0.99,
                        "length": 6.4e-8,
                    },
                    "Q27": {
                        "T1": 35.0,
                        "T2": 50.0,
                        "fidelity": 0.99,
                        "length": 6.4e-8,
                    },
                },
                "couplers_info": {
                    "C0": {
                        "qubits_index": [20, 27],
                        "fidelity": 0.98,
                        "length": 2.24e-7,
                    }
                },
            },
            target="quafu:Shenglian",
            qubits=(20, 27),
        )
        for twin in (reference, current)
    )
    current_without_readout = without_readout[-1]

    assessment = fq.twin.TwinFreshnessPolicy(maximum_readout_tv_drift=0.02).assess(
        current_without_readout,
        calibration_history=fq.twin.build_calibration_history(without_readout),
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )

    assert assessment.status == "insufficient_evidence"
    assert assessment.reasons == ("readout_drift_unavailable",)


def test_incompatible_target_snapshot_and_time_fail_closed():
    _, current, calibration, validation = _history_fixture()
    other = _twin(
        captured_at="2026-08-14T11:30:00+08:00",
        changed=True,
        backend="Baihua",
    )

    target_mismatch = _policy().assess(
        other,
        calibration_history=calibration,
        validation_history=validation,
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )
    early = _policy().assess(
        current,
        calibration_history=calibration,
        validation_history=validation,
        as_of=datetime(2026, 8, 14, 2, 0, tzinfo=timezone.utc),
    )

    assert target_mismatch.status == "incompatible_history"
    assert set(target_mismatch.reasons) == {
        "calibration_history_target_mismatch",
        "validation_history_target_mismatch",
    }
    assert early.status == "incompatible_history"
    assert set(early.reasons) == {
        "as_of_precedes_calibration_snapshot",
        "as_of_precedes_validation_snapshot",
    }


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({}, ValueError, "at least one threshold"),
        (
            {"maximum_calibration_age": timedelta(0)},
            ValueError,
            "finite and positive",
        ),
        (
            {"maximum_validation_age": 60},
            TypeError,
            "datetime.timedelta",
        ),
        (
            {"maximum_readout_tv_drift": 1.1},
            ValueError,
            r"\[0, 1\]",
        ),
        (
            {"maximum_relative_t1_drift": -0.1},
            ValueError,
            "non-negative",
        ),
    ],
)
def test_policy_rejects_invalid_thresholds(kwargs, error, message):
    with pytest.raises(error, match=message):
        fq.twin.TwinFreshnessPolicy(**kwargs)


def test_policy_requires_timezone_and_public_input_types():
    _, current, calibration, validation = _history_fixture()
    policy = _policy()

    with pytest.raises(ValueError, match="timezone"):
        policy.assess(
            current,
            calibration_history=calibration,
            validation_history=validation,
            as_of=datetime(2026, 8, 14, 4, 0),
        )
    with pytest.raises(TypeError, match="QPUDigitalTwin"):
        policy.assess(object(), as_of=datetime.now(timezone.utc))
    with pytest.raises(TypeError, match="TwinCalibrationHistory"):
        policy.assess(
            current,
            calibration_history=object(),
            as_of=datetime.now(timezone.utc),
        )
    with pytest.raises(TypeError, match="TwinValidationHistory"):
        policy.assess(
            current,
            validation_history=object(),
            as_of=datetime.now(timezone.utc),
        )


def test_freshness_records_are_immutable_and_have_no_routing_authority():
    _, current, calibration, validation = _history_fixture()
    policy = _policy()
    assessment = policy.assess(
        current,
        calibration_history=calibration,
        validation_history=validation,
        as_of=datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc),
    )

    assert policy.__dataclass_params__.frozen is True
    assert assessment.__dataclass_params__.frozen is True
    assert not hasattr(assessment, "routing_authorized")
    with pytest.raises(dataclasses.FrozenInstanceError):
        assessment.status = "refresh_due"
