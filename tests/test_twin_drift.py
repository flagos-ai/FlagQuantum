"""Provider-neutral Twin calibration-drift comparison tests."""

from __future__ import annotations

import json

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration


def _chip_info(*, captured_at="2026-08-14 10:30:00", changed=False):
    return {
        "calibration_time": captured_at,
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
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


def _twin(*, captured_at="2026-08-14 10:30:00", changed=False, backend="Shenglian"):
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
        _chip_info(captured_at=captured_at, changed=changed),
        target=f"quafu:{backend}",
        qubits=(20, 27),
        readout_confusion_matrices=readout,
    )


def test_compare_calibrations_reports_per_qubit_and_gate_drift():
    reference = _twin()
    current = _twin(captured_at="2026-08-14 11:30:00", changed=True)

    drift = fq.twin.compare_calibrations(reference, current)

    assert drift.provider == "quafu"
    assert drift.backend_name == "Shenglian"
    assert drift.physical_qubits == (20, 27)
    assert drift.elapsed_seconds == 3600
    assert drift.qubit_drifts[0].physical_qubit == 20
    assert drift.qubit_drifts[0].relative_t1_delta == pytest.approx(0.1)
    assert drift.qubit_drifts[0].relative_t2_delta == pytest.approx(-0.1)
    assert drift.qubit_drifts[0].maximum_readout_row_tv_distance == pytest.approx(0.03)
    assert drift.qubit_drifts[1].maximum_readout_row_tv_distance == 0
    assert drift.maximum_relative_t1_change == pytest.approx(0.1)
    assert drift.maximum_relative_t2_change == pytest.approx(0.1)
    assert drift.maximum_readout_tv_distance == pytest.approx(0.03)
    assert drift.maximum_relative_gate_duration_change == pytest.approx(0.1)
    assert drift.channel_model_changed is True
    assert drift.has_observed_drift is True
    assert any(
        item.gate_name == "cx" and item.physical_qubits == (20, 27)
        for item in drift.gate_duration_drifts
    )


def test_compare_same_calibration_has_no_observed_drift():
    twin = _twin()

    drift = fq.twin.compare_calibrations(twin, twin)

    assert drift.elapsed_seconds == 0
    assert drift.maximum_relative_t1_change == 0
    assert drift.maximum_relative_t2_change == 0
    assert drift.maximum_readout_tv_distance == 0
    assert drift.maximum_relative_gate_duration_change == 0
    assert drift.channel_model_changed is False
    assert drift.has_observed_drift is False


def test_drift_report_is_json_ready_and_identity_bound():
    drift = fq.twin.compare_calibrations(
        _twin(), _twin(captured_at="2026-08-14 11:30:00", changed=True)
    )

    payload = drift.to_dict()

    assert payload["reference_snapshot_identity"] == drift.reference_snapshot_identity
    assert payload["current_snapshot_identity"] == drift.current_snapshot_identity
    assert payload["qubit_drifts"][0]["physical_qubit"] == 20
    assert payload["has_observed_drift"] is True
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("reference", "current", "message"),
    [
        (_twin(), _twin(backend="Baihua"), "same target and mapping"),
        (
            _twin(captured_at="2026-08-14 11:30:00"),
            _twin(captured_at="2026-08-14 10:30:00"),
            "predates",
        ),
    ],
)
def test_compare_calibrations_rejects_incomparable_twins(reference, current, message):
    with pytest.raises(ValueError, match=message):
        fq.twin.compare_calibrations(reference, current)


def test_compare_calibrations_requires_twins():
    with pytest.raises(TypeError, match="QPUDigitalTwin"):
        fq.twin.compare_calibrations(object(), _twin())


def test_compare_calibrations_rejects_changed_readout_availability():
    reference = _twin()
    current = fq.twin.from_quafu_chip_info(
        _chip_info(captured_at="2026-08-14 11:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )

    with pytest.raises(ValueError, match="readout calibration availability"):
        fq.twin.compare_calibrations(reference, current)
