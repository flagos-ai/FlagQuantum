"""Aligned Twin calibration and validation-history scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import flagquantum as fq

_CIRCUIT_IDENTITY = fq.Circuit(2).h(0).cx(0, 1).to_ir().content_hash


def _twin(*, captured_at: str, t1: float, backend: str = "Shenglian"):
    return fq.twin.from_quafu_chip_info(
        {
            "calibration_time": captured_at,
            "basis_gates": ["h", "cx"],
            "qubits_info": {
                "Q20": {
                    "T1": t1,
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
        target=f"quafu:{backend}",
        qubits=(20, 27),
    )


def _series(
    twin,
    *,
    label: str,
    twin_distance: float,
    ideal_distance: float,
    repeatability_distance: float,
):
    return fq.twin.TwinValidationSeries(
        snapshot_identity=twin.snapshot.identity,
        circuit_identity=_CIRCUIT_IDENTITY,
        physical_qubits=(20, 27),
        report_identities=tuple(
            hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
            for index in range(2)
        ),
        repetitions=2,
        total_shots=2048,
        mean_twin_hardware_total_variation=twin_distance,
        maximum_twin_hardware_total_variation=twin_distance,
        mean_ideal_hardware_total_variation=ideal_distance,
        maximum_ideal_hardware_total_variation=ideal_distance,
        mean_hardware_repeatability_total_variation=repeatability_distance,
        maximum_hardware_repeatability_total_variation=repeatability_distance,
        simultaneous_finite_shot_tv_radius=0.05,
        verified_tv_error_bound=twin_distance + 0.05,
        confidence_level=0.95,
        supported_operations=("cx", "h", "measure"),
        maximum_instruction_count=2,
    )


def _histories():
    twins = (
        _twin(captured_at="2026-08-14 10:30:00", t1=40.0),
        _twin(captured_at="2026-08-14 11:30:00", t1=44.0),
        _twin(captured_at="2026-08-14 13:30:00", t1=48.0),
    )
    series = (
        _series(
            twins[0],
            label="state-01",
            twin_distance=0.08,
            ideal_distance=0.12,
            repeatability_distance=0.03,
        ),
        _series(
            twins[1],
            label="state-02",
            twin_distance=0.06,
            ideal_distance=0.11,
            repeatability_distance=0.04,
        ),
        _series(
            twins[2],
            label="state-03",
            twin_distance=0.07,
            ideal_distance=0.13,
            repeatability_distance=0.02,
        ),
    )
    return (
        fq.twin.build_calibration_history(twins),
        fq.twin.build_validation_history(tuple(zip(twins, series, strict=True))),
    )


def test_align_histories_reports_device_and_validation_changes_separately():
    calibration, validation = _histories()

    evolution = fq.twin.align_histories(calibration, validation)

    assert evolution.observation_count == 3
    assert evolution.twin_qpu_agreement_changes == pytest.approx((0.02, -0.01))
    assert evolution.ideal_qpu_agreement_changes == pytest.approx((0.01, -0.02))
    assert evolution.qpu_repeatability_changes == pytest.approx((-0.01, 0.02))
    assert evolution.verified_tv_error_bound_changes == pytest.approx((-0.02, 0.01))
    assert evolution.latest_calibration_drift is calibration.latest_interval_drift
    assert evolution.latest_twin_agreement_change == pytest.approx(-0.01)
    assert evolution.latest_ideal_agreement_change == pytest.approx(-0.02)
    assert evolution.latest_qpu_repeatability_change == pytest.approx(0.02)
    assert evolution.latest_verified_bound_change == pytest.approx(0.01)


def test_evolution_history_is_json_ready_without_causal_claims():
    evolution = fq.twin.align_histories(*_histories())

    payload = evolution.to_dict()

    assert payload["observation_count"] == 3
    assert payload["twin_qpu_agreement_changes"] == pytest.approx([0.02, -0.01])
    assert payload["verified_tv_error_bound_changes"] == pytest.approx([-0.02, 0.01])
    assert "cause" not in payload
    json.dumps(payload, allow_nan=False)


def test_align_histories_preserves_unavailable_repeatability():
    calibration, validation = _histories()
    first = replace(
        validation.validation_series[0],
        report_identities=(validation.validation_series[0].report_identities[0],),
        repetitions=1,
        total_shots=1024,
        mean_hardware_repeatability_total_variation=None,
        maximum_hardware_repeatability_total_variation=None,
    )
    validation = replace(
        validation,
        validation_series=(first, *validation.validation_series[1:]),
    )

    evolution = fq.twin.align_histories(calibration, validation)

    assert evolution.qpu_repeatability_changes[0] is None
    assert evolution.qpu_repeatability_changes[1] == pytest.approx(0.02)


def test_align_histories_rejects_wrong_types():
    calibration, validation = _histories()

    with pytest.raises(TypeError, match="TwinCalibrationHistory"):
        fq.twin.align_histories(object(), validation)
    with pytest.raises(TypeError, match="TwinValidationHistory"):
        fq.twin.align_histories(calibration, object())


def test_align_histories_rejects_snapshot_or_time_mismatch():
    calibration, validation = _histories()
    other_twins = (
        _twin(captured_at="2026-08-14 10:30:00", t1=41.0),
        _twin(captured_at="2026-08-14 11:30:00", t1=45.0),
        _twin(captured_at="2026-08-14 13:30:00", t1=49.0),
    )

    with pytest.raises(ValueError, match="same snapshot identities"):
        fq.twin.align_histories(
            fq.twin.build_calibration_history(other_twins),
            validation,
        )

    changed_times = replace(
        validation,
        captured_at=(
            validation.captured_at[0],
            "2026-08-14 12:30:00+08:00",
            validation.captured_at[2],
        ),
    )
    with pytest.raises(ValueError, match="same captured_at"):
        fq.twin.align_histories(calibration, changed_times)


def test_align_histories_rejects_target_mismatch():
    calibration, validation = _histories()
    changed_target = replace(validation, backend_name="Baihua")

    with pytest.raises(ValueError, match="same target and mapping"):
        fq.twin.align_histories(calibration, changed_target)
