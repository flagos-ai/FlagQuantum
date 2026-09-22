"""Longitudinal connected-region Twin validation scenarios."""

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


def _circuits():
    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).h(1).cx(1, 2),
    )


def _evaluation(
    region_twin: fq.twin.TwinRegionModel,
    label: str,
    *,
    distance: float,
) -> fq.twin.TwinRegionSuiteEvaluation:
    confidence = 0.95
    per_circuit_confidence = 1.0 - (1.0 - confidence) / 2.0
    validation_series = []
    for circuit_index, circuit in enumerate(_circuits(), start=1):
        maximum_distance = distance + circuit_index / 1000.0
        radius = 0.08 + circuit_index / 1000.0
        validation_series.append(
            fq.twin.TwinValidationSeries(
                snapshot_identity=region_twin.twin.snapshot.identity,
                circuit_identity=circuit.to_ir().content_hash,
                physical_qubits=(20, 27, 34),
                report_identities=(
                    _digest(f"{label}-{circuit_index}-1"),
                    _digest(f"{label}-{circuit_index}-2"),
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
        suite_identity=_digest(f"suite-{label}"),
        region_identity=region_twin.identity,
        snapshot_identity=region_twin.twin.snapshot.identity,
        physical_qubits=(20, 27, 34),
        validation_series=tuple(validation_series),
        directed_couplers=((20, 27), (27, 34)),
        maximum_circuit_depth=3,
        confidence_level=confidence,
    )


def _observations():
    reference = _region_twin("2026-08-14 10:30:00")
    current = _region_twin(
        "2026-08-15 10:30:00",
        fidelity_shift=-0.001,
    )
    return (
        (reference, _evaluation(reference, "state-01", distance=0.03)),
        (current, _evaluation(current, "state-02", distance=0.04)),
    )


def test_region_validation_history_tracks_fixed_suite_over_time() -> None:
    observations = _observations()

    history = fq.twin.build_region_validation_history(observations)

    assert history.provider == "quafu"
    assert history.backend_name == "Shenglian"
    assert history.physical_qubits == (20, 27, 34)
    assert history.directed_couplers == ((20, 27), (27, 34))
    assert history.circuit_identities == tuple(
        circuit.to_ir().content_hash for circuit in _circuits()
    )
    assert history.observation_count == 2
    assert history.mean_twin_qpu_agreements == pytest.approx((0.97, 0.96))
    assert history.mean_ideal_qpu_agreements == pytest.approx((0.87, 0.86))
    assert history.mean_qpu_repeatabilities == pytest.approx((0.99, 0.99))
    assert history.confidence_levels == pytest.approx((0.95, 0.95))
    assert history.task_counts == (4, 4)
    assert history.total_shots == (4096, 4096)


def test_region_validation_history_append_is_immutable() -> None:
    history = fq.twin.build_region_validation_history(_observations())
    later = _region_twin("2026-08-16 10:30:00", fidelity_shift=-0.002)
    later_evaluation = _evaluation(later, "state-03", distance=0.05)

    updated = history.append(later, later_evaluation)

    assert history.observation_count == 2
    assert updated.observation_count == 3
    assert updated.snapshot_identities[-1] == later.twin.snapshot.identity
    assert updated.mean_twin_qpu_agreements[-1] == pytest.approx(0.95)


def test_region_validation_history_round_trip_is_private_and_create_once(
    tmp_path,
) -> None:
    history = fq.twin.build_region_validation_history(_observations())
    destination = tmp_path / "region-validation-history.json"

    fq.twin.dump_region_validation_history(history, destination)
    original = destination.read_bytes()
    restored = fq.twin.load_region_validation_history(destination)
    fq.twin.dump_region_validation_history(history, destination)

    assert restored == history
    assert destination.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o600


def test_region_validation_history_rejects_changed_circuit_suite() -> None:
    first, second = _observations()
    reordered = replace(
        second[1],
        validation_series=tuple(reversed(second[1].validation_series)),
    )

    with pytest.raises(ValueError, match="ordered circuit suite"):
        fq.twin.build_region_validation_history((first, (second[0], reordered)))


def test_region_validation_history_rejects_changed_topology() -> None:
    first, second = _observations()
    reversed_region = replace(
        second[0],
        region=replace(
            second[0].region,
            directed_couplers=((27, 20), (34, 27)),
        ),
    )
    rebound = replace(second[1], region_identity=reversed_region.identity)

    with pytest.raises(ValueError, match="topology"):
        fq.twin.build_region_validation_history((first, (reversed_region, rebound)))


def test_region_validation_history_rejects_reused_hardware_reports() -> None:
    first, second = _observations()
    reused_series = tuple(
        replace(
            current,
            report_identities=reference.report_identities,
        )
        for reference, current in zip(
            first[1].validation_series,
            second[1].validation_series,
            strict=True,
        )
    )
    reused = replace(second[1], validation_series=reused_series)

    with pytest.raises(ValueError, match="distinct hardware reports"):
        fq.twin.build_region_validation_history((first, (second[0], reused)))


def test_region_validation_history_rejects_noncanonical_file(tmp_path) -> None:
    history = fq.twin.build_region_validation_history(_observations())
    payload = history.to_dict()
    payload["mean_twin_qpu_agreements"][0] = 0.123
    destination = tmp_path / "tampered.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Twin region-validation history"):
        fq.twin.load_region_validation_history(destination)


def test_region_validation_history_requires_chronological_observations() -> None:
    observations = _observations()

    with pytest.raises(ValueError, match="strictly increasing"):
        fq.twin.build_region_validation_history(tuple(reversed(observations)))
