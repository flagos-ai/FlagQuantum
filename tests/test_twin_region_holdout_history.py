"""Longitudinal connected-region Twin holdout scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _chip_info(captured_at: str, *, fidelity_shift: float = 0.0):
    return {
        "calibration_time": captured_at,
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {
                "T1": 41.0,
                "T2": 61.0,
                "fidelity": 0.997 + fidelity_shift,
                "length": 6.4e-8,
            },
            "Q27": {
                "T1": 40.0,
                "T2": 60.0,
                "fidelity": 0.998 + fidelity_shift,
                "length": 6.4e-8,
            },
            "Q34": {
                "T1": 42.0,
                "T2": 62.0,
                "fidelity": 0.996 + fidelity_shift,
                "length": 6.4e-8,
            },
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 0.985 + fidelity_shift,
                "length": 2.24e-7,
            },
            "C1": {
                "qubits_index": [27, 34],
                "fidelity": 0.984 + fidelity_shift,
                "length": 2.24e-7,
            },
        },
    }


def _cell(chip_info, qubits: tuple[int, int]):
    twin = fq.twin.from_quafu_chip_info(
        chip_info,
        target="quafu:Shenglian",
        qubits=qubits,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    support = fq.twin.TwinCircuitSupport(
        evidence=fq.twin.TwinEvidenceEnvelope(
            snapshot_identity=twin.snapshot.identity,
            physical_qubits=qubits,
            supported_operations=("h", "cx"),
            maximum_instruction_count=8,
            verified_circuit_identities=(circuit.to_ir().content_hash,),
            evidence_identity=_digest(f"cell-{twin.snapshot.identity}"),
            verified_tv_error_bound=0.04,
            estimated_tv_error_bound=None,
            confidence_level=0.95,
        ),
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=4,
    )
    return twin, support


def _region_twin(
    captured_at: str,
    *,
    fidelity_shift: float = 0.0,
) -> fq.twin.TwinRegionModel:
    chip_info = _chip_info(captured_at, fidelity_shift=fidelity_shift)
    return fq.twin.compose_region_twin(
        (_cell(chip_info, (20, 27)), _cell(chip_info, (27, 34)))
    )


def _reference_circuits():
    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).h(1).cx(1, 2),
    )


def _holdout_circuits():
    return (
        fq.Circuit(3).h(0).cx(0, 1),
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2).h(1),
    )


def _suite_evaluation(
    region_twin: fq.twin.TwinRegionModel,
    label: str,
    group: str,
    circuits,
    *,
    distance: float,
) -> fq.twin.TwinRegionSuiteEvaluation:
    confidence = 0.975
    per_circuit_confidence = 1.0 - (1.0 - confidence) / len(circuits)
    series = []
    for circuit_index, circuit in enumerate(circuits, start=1):
        radius = 0.08 + circuit_index / 1000.0
        maximum_distance = distance + circuit_index / 1000.0
        series.append(
            fq.twin.TwinValidationSeries(
                snapshot_identity=region_twin.twin.snapshot.identity,
                circuit_identity=circuit.to_ir().content_hash,
                physical_qubits=(20, 27, 34),
                report_identities=(
                    _digest(f"{label}-{group}-{circuit_index}-1"),
                    _digest(f"{label}-{group}-{circuit_index}-2"),
                ),
                repetitions=2,
                total_shots=2048,
                mean_twin_hardware_total_variation=distance,
                maximum_twin_hardware_total_variation=maximum_distance,
                mean_ideal_hardware_total_variation=distance + 0.1,
                maximum_ideal_hardware_total_variation=distance + 0.11,
                mean_hardware_repeatability_total_variation=0.01,
                maximum_hardware_repeatability_total_variation=0.015,
                simultaneous_finite_shot_tv_radius=radius,
                verified_tv_error_bound=maximum_distance + radius,
                confidence_level=per_circuit_confidence,
                supported_operations=("cx", "h", "measure"),
                maximum_instruction_count=len(circuit.to_ir().instructions),
            )
        )
    return fq.twin.TwinRegionSuiteEvaluation(
        suite_identity=_digest(f"suite-{label}-{group}"),
        region_identity=region_twin.identity,
        snapshot_identity=region_twin.twin.snapshot.identity,
        physical_qubits=(20, 27, 34),
        validation_series=tuple(series),
        directed_couplers=((20, 27), (27, 34)),
        maximum_circuit_depth=3,
        confidence_level=confidence,
    )


def _evaluation(
    region_twin: fq.twin.TwinRegionModel,
    label: str,
    *,
    reference_distance: float,
    holdout_distance: float,
) -> fq.twin.TwinRegionHoldoutEvaluation:
    return fq.twin.TwinRegionHoldoutEvaluation(
        study_identity=_digest(f"study-{label}"),
        reference_evaluation=_suite_evaluation(
            region_twin,
            label,
            "reference",
            _reference_circuits(),
            distance=reference_distance,
        ),
        holdout_evaluation=_suite_evaluation(
            region_twin,
            label,
            "holdout",
            _holdout_circuits(),
            distance=holdout_distance,
        ),
        confidence_level=0.95,
    )


def _observations():
    reference = _region_twin("2026-08-14 10:30:00")
    current = _region_twin("2026-08-15 10:30:00", fidelity_shift=-0.001)
    return (
        (
            reference,
            _evaluation(
                reference,
                "state-01",
                reference_distance=0.03,
                holdout_distance=0.04,
            ),
        ),
        (
            current,
            _evaluation(
                current,
                "state-02",
                reference_distance=0.04,
                holdout_distance=0.06,
            ),
        ),
    )


def test_region_holdout_history_tracks_fixed_study_over_time() -> None:
    history = fq.twin.build_region_holdout_history(_observations())

    assert history.provider == "quafu"
    assert history.backend_name == "Shenglian"
    assert history.physical_qubits == (20, 27, 34)
    assert history.directed_couplers == ((20, 27), (27, 34))
    assert history.observation_count == 2
    assert history.reference_twin_qpu_agreements == pytest.approx((0.97, 0.96))
    assert history.holdout_twin_qpu_agreements == pytest.approx((0.96, 0.94))
    assert history.holdout_tv_error_increases == pytest.approx((0.01, 0.02))
    assert history.reference_ideal_qpu_agreements == pytest.approx((0.87, 0.86))
    assert history.holdout_ideal_qpu_agreements == pytest.approx((0.86, 0.84))
    assert history.reference_qpu_repeatabilities == pytest.approx((0.99, 0.99))
    assert history.holdout_qpu_repeatabilities == pytest.approx((0.99, 0.99))
    assert history.confidence_levels == pytest.approx((0.95, 0.95))
    assert history.task_counts == (8, 8)
    assert history.total_shots == (8192, 8192)


def test_region_holdout_history_append_is_immutable() -> None:
    history = fq.twin.build_region_holdout_history(_observations())
    later = _region_twin("2026-08-16 10:30:00", fidelity_shift=-0.002)
    evaluation = _evaluation(
        later,
        "state-03",
        reference_distance=0.05,
        holdout_distance=0.07,
    )

    updated = history.append(later, evaluation)

    assert history.observation_count == 2
    assert updated.observation_count == 3
    assert updated.holdout_twin_qpu_agreements[-1] == pytest.approx(0.93)


def test_region_holdout_history_round_trip_is_private_and_create_once(
    tmp_path,
) -> None:
    history = fq.twin.build_region_holdout_history(_observations())
    destination = tmp_path / "region-holdout-history.json"

    fq.twin.dump_region_holdout_history(history, destination)
    original = destination.read_bytes()
    restored = fq.twin.load_region_holdout_history(destination)
    fq.twin.dump_region_holdout_history(history, destination)

    assert restored == history
    assert destination.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o600


def test_region_holdout_history_rejects_changed_ordered_holdout_suite() -> None:
    first, second = _observations()
    reordered_suite = replace(
        second[1].holdout_evaluation,
        validation_series=tuple(
            reversed(second[1].holdout_evaluation.validation_series)
        ),
    )
    reordered = replace(second[1], holdout_evaluation=reordered_suite)

    with pytest.raises(ValueError, match="ordered holdout circuit suite"):
        fq.twin.build_region_holdout_history((first, (second[0], reordered)))


def test_region_holdout_history_rejects_changed_experiment_design() -> None:
    first, second = _observations()
    changed_series = replace(
        second[1].reference_evaluation.validation_series[0],
        total_shots=4096,
    )
    changed_suite = replace(
        second[1].reference_evaluation,
        validation_series=(
            changed_series,
            second[1].reference_evaluation.validation_series[1],
        ),
    )
    changed = replace(second[1], reference_evaluation=changed_suite)

    with pytest.raises(ValueError, match="one experiment design"):
        fq.twin.build_region_holdout_history((first, (second[0], changed)))


def test_region_holdout_history_rejects_changed_topology() -> None:
    first, second = _observations()
    reversed_region = replace(
        second[0],
        region=replace(
            second[0].region,
            directed_couplers=((27, 20), (34, 27)),
        ),
    )
    rebound = replace(
        second[1],
        reference_evaluation=replace(
            second[1].reference_evaluation,
            region_identity=reversed_region.identity,
        ),
        holdout_evaluation=replace(
            second[1].holdout_evaluation,
            region_identity=reversed_region.identity,
        ),
    )

    with pytest.raises(ValueError, match="topology"):
        fq.twin.build_region_holdout_history((first, (reversed_region, rebound)))


def test_region_holdout_history_rejects_reused_hardware_reports() -> None:
    first, second = _observations()
    reused_series = replace(
        second[1].holdout_evaluation.validation_series[0],
        report_identities=(
            first[1].reference_evaluation.validation_series[0].report_identities
        ),
    )
    reused_suite = replace(
        second[1].holdout_evaluation,
        validation_series=(
            reused_series,
            second[1].holdout_evaluation.validation_series[1],
        ),
    )
    reused = replace(second[1], holdout_evaluation=reused_suite)

    with pytest.raises(ValueError, match="distinct hardware reports"):
        fq.twin.build_region_holdout_history((first, (second[0], reused)))


def test_region_holdout_history_rejects_noncanonical_file(tmp_path) -> None:
    history = fq.twin.build_region_holdout_history(_observations())
    payload = history.to_dict()
    payload["holdout_twin_qpu_agreements"][0] = 0.123
    destination = tmp_path / "tampered-region-holdout-history.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Twin region holdout history"):
        fq.twin.load_region_holdout_history(destination)


def test_region_holdout_history_requires_chronological_observations() -> None:
    observations = _observations()

    with pytest.raises(ValueError, match="strictly increasing"):
        fq.twin.build_region_holdout_history(tuple(reversed(observations)))
