"""Longitudinal validation-history scenarios for frozen QPU Twins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import flagquantum as fq

_CIRCUIT_IDENTITY = fq.Circuit(2).h(0).cx(0, 1).to_ir().content_hash


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


def _series(
    twin,
    *,
    label: str,
    twin_distance: float,
    ideal_distance: float,
    repeatability_distance: float = 0.03,
    circuit_identity: str = _CIRCUIT_IDENTITY,
    operations: tuple[str, ...] = ("cx", "h", "measure"),
):
    report_identities = tuple(
        hashlib.sha256(f"{label}-{index}".encode()).hexdigest() for index in range(2)
    )
    return fq.twin.TwinValidationSeries(
        snapshot_identity=twin.snapshot.identity,
        circuit_identity=circuit_identity,
        physical_qubits=(20, 27),
        report_identities=report_identities,
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
        supported_operations=operations,
        maximum_instruction_count=2,
    )


def _observations():
    first = _twin(captured_at="2026-08-14 10:30:00", t1=40.0)
    second = _twin(captured_at="2026-08-14 11:30:00", t1=44.0)
    third = _twin(captured_at="2026-08-14 13:30:00", t1=48.0)
    return (
        (
            first,
            _series(first, label="state-01", twin_distance=0.08, ideal_distance=0.12),
        ),
        (
            second,
            _series(second, label="state-02", twin_distance=0.06, ideal_distance=0.11),
        ),
        (
            third,
            _series(third, label="state-03", twin_distance=0.07, ideal_distance=0.13),
        ),
    )


def test_build_validation_history_preserves_distinct_time_series():
    observations = _observations()
    history = fq.twin.build_validation_history(observations)

    assert history.provider == "quafu"
    assert history.backend_name == "Shenglian"
    assert history.physical_qubits == (20, 27)
    assert history.circuit_identity == _CIRCUIT_IDENTITY
    assert history.observation_count == 3
    assert history.snapshot_identities == tuple(
        twin.snapshot.identity for twin, _ in observations
    )
    assert history.mean_twin_qpu_agreements == pytest.approx((0.92, 0.94, 0.93))
    assert history.mean_ideal_qpu_agreements == pytest.approx((0.88, 0.89, 0.87))
    assert history.mean_qpu_repeatabilities == pytest.approx((0.97, 0.97, 0.97))
    assert history.simultaneous_finite_shot_tv_radii == pytest.approx(
        (0.05, 0.05, 0.05)
    )
    assert history.confidence_levels == pytest.approx((0.95, 0.95, 0.95))
    assert history.verified_tv_error_bounds == pytest.approx((0.13, 0.11, 0.12))


def test_validation_history_is_json_ready_without_collapsing_metrics():
    history = fq.twin.build_validation_history(_observations())

    payload = history.to_dict()

    assert payload["observation_count"] == 3
    assert payload["mean_twin_qpu_agreements"] == pytest.approx([0.92, 0.94, 0.93])
    assert payload["mean_ideal_qpu_agreements"] == pytest.approx([0.88, 0.89, 0.87])
    assert payload["mean_qpu_repeatabilities"] == pytest.approx([0.97, 0.97, 0.97])
    assert payload["simultaneous_finite_shot_tv_radii"] == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    assert payload["confidence_levels"] == pytest.approx([0.95, 0.95, 0.95])
    assert payload["validation_series"][0]["confidence_level"] == 0.95
    json.dumps(payload, allow_nan=False)


def test_validation_history_rejects_snapshot_mismatch():
    first, second = _observations()[:2]
    with pytest.raises(ValueError, match="does not match its Twin snapshot"):
        fq.twin.build_validation_history((first, (second[0], first[1])))


def test_validation_history_rejects_changed_target_or_chronology():
    first = _observations()[0]
    other = _twin(captured_at="2026-08-14 11:30:00", t1=44.0, backend="Baihua")
    other_series = _series(
        other, label="other", twin_distance=0.08, ideal_distance=0.12
    )
    with pytest.raises(ValueError, match="one target and mapping"):
        fq.twin.build_validation_history((first, (other, other_series)))

    observations = _observations()
    with pytest.raises(ValueError, match="strictly increasing"):
        fq.twin.build_validation_history((observations[1], observations[0]))


def test_validation_history_rejects_changed_circuit_or_structure():
    observations = _observations()
    twin = observations[1][0]
    changed_circuit = _series(
        twin,
        label="changed-circuit",
        twin_distance=0.06,
        ideal_distance=0.11,
        circuit_identity="a" * 64,
    )
    with pytest.raises(ValueError, match="one circuit and mapping"):
        fq.twin.build_validation_history((observations[0], (twin, changed_circuit)))

    changed_structure = _series(
        twin,
        label="changed-structure",
        twin_distance=0.06,
        ideal_distance=0.11,
        operations=("h", "measure"),
    )
    with pytest.raises(ValueError, match="structures differ"):
        fq.twin.build_validation_history((observations[0], (twin, changed_structure)))


def test_validation_history_rejects_reused_hardware_reports():
    observations = _observations()
    first_series = observations[0][1]
    second = observations[1][0]
    reused = replace(first_series, snapshot_identity=second.snapshot.identity)

    with pytest.raises(ValueError, match="distinct hardware reports"):
        fq.twin.build_validation_history((observations[0], (second, reused)))


@pytest.mark.parametrize(
    ("observations", "error", "message"),
    [
        ((), ValueError, "at least two"),
        ((_observations()[0],), ValueError, "at least two"),
        ((object(), object()), TypeError, "pairs"),
        (((object(), object()), _observations()[1]), TypeError, "TwinValidationSeries"),
    ],
)
def test_validation_history_rejects_invalid_observations(observations, error, message):
    with pytest.raises(error, match=message):
        fq.twin.build_validation_history(observations)
