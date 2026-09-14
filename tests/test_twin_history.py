"""Chronological Twin calibration-history tests."""

from __future__ import annotations

import json

import pytest

import flagquantum as fq


def _twin(*, captured_at: str, t1: float, backend: str = "Shenglian"):
    chip_info = {
        "calibration_time": captured_at,
        "basis_gates": ["h", "cx"],
        "qubits_info": {
            "Q20": {"T1": t1, "T2": 60.0, "fidelity": 0.99, "length": 6.4e-8},
            "Q27": {"T1": 35.0, "T2": 50.0, "fidelity": 0.99, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {"qubits_index": [20, 27], "fidelity": 0.98, "length": 2.24e-7}
        },
    }
    return fq.twin.from_quafu_chip_info(
        chip_info,
        target=f"quafu:{backend}",
        qubits=(20, 27),
    )


def _series():
    return (
        _twin(captured_at="2026-08-14 10:30:00", t1=40.0),
        _twin(captured_at="2026-08-14 11:30:00", t1=44.0),
        _twin(captured_at="2026-08-14 13:30:00", t1=48.0),
    )


def test_build_calibration_history_reports_cumulative_and_interval_drift():
    twins = _series()
    history = fq.twin.build_calibration_history(twins)

    assert history.provider == "quafu"
    assert history.backend_name == "Shenglian"
    assert history.physical_qubits == (20, 27)
    assert history.snapshot_identities == tuple(
        twin.snapshot.identity for twin in twins
    )
    assert history.observation_count == 3
    assert len(history.baseline_drifts) == 2
    assert len(history.interval_drifts) == 2
    assert history.baseline_drifts[1].qubit_drifts[
        0
    ].relative_t1_delta == pytest.approx(0.2)
    assert history.interval_drifts[1].qubit_drifts[
        0
    ].relative_t1_delta == pytest.approx(48 / 44 - 1)
    assert history.latest_baseline_drift is history.baseline_drifts[-1]
    assert history.latest_interval_drift is history.interval_drifts[-1]


def test_calibration_history_is_json_ready():
    history = fq.twin.build_calibration_history(_series())

    payload = history.to_dict()

    assert payload["observation_count"] == 3
    assert len(payload["baseline_drifts"]) == 2
    assert len(payload["interval_drifts"]) == 2
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("twins", "error", "message"),
    [
        ((), ValueError, "at least two"),
        ((_series()[0],), ValueError, "at least two"),
        ((_series()[0], object()), TypeError, "QPUDigitalTwin"),
        ((_series()[0], _series()[0]), ValueError, "identities must be unique"),
        ((_series()[1], _series()[0]), ValueError, "strictly increasing"),
        (
            (
                _series()[0],
                _twin(captured_at="2026-08-14 11:30:00", t1=44.0, backend="Baihua"),
            ),
            ValueError,
            "same target and mapping",
        ),
    ],
)
def test_build_calibration_history_rejects_invalid_series(twins, error, message):
    with pytest.raises(error, match=message):
        fq.twin.build_calibration_history(twins)
