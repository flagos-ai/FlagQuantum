"""Scenarios for immutable Twin candidates fitted from probe estimates."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
)

pytestmark = pytest.mark.integration


def _incumbent() -> fq.twin.QPUDigitalTwin:
    profile = DeviceNoiseProfile(
        qubits=(
            QubitNoiseCalibration(0, t1=40_000.0, t2=60_000.0),
            QubitNoiseCalibration(1, t1=35_000.0, t2=50_000.0),
        ),
        gate_durations=(
            GateDuration("h", 64.0, wires=(0,)),
            GateDuration("x", 64.0, wires=(0,)),
            GateDuration("h", 64.0, wires=(1,)),
            GateDuration("x", 64.0, wires=(1,)),
            GateDuration("cx", 224.0, wires=(0, 1)),
            GateDuration("cx", 224.0, wires=(1, 0)),
        ),
        source="synthetic-calibration",
        captured_at="2026-08-14T10:30:00+08:00",
        time_unit="ns",
    )
    return fq.twin.from_noise_model(
        NoiseModel.from_device_profile(profile),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )


def _estimate(
    incumbent: fq.twin.QPUDigitalTwin,
) -> fq.twin.TwinParameterEstimate:
    parameters = (
        ("readout_bit_flip[Q20]", 0.01),
        ("readout_bit_flip[Q27]", 0.02),
        ("single_qubit_pauli[Q20]", 0.03),
        ("single_qubit_pauli[Q27]", 0.04),
        ("two_qubit_pauli[Q20->Q27]", 0.05),
    )
    return fq.twin.TwinParameterEstimate(
        probe_bank_identity="b" * 64,
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        parameter_family="local_pauli_readout",
        parameter_values=parameters,
        confidence_intervals=tuple(
            (name, max(0.0, value - 0.005), value + 0.005) for name, value in parameters
        ),
        identifiability_status="identifiable",
        unidentified_parameters=(),
        boundary_parameters=(),
        hardware_report_identities=tuple(character * 64 for character in "cdef"),
        task_ids=tuple(f"task-{index}" for index in range(4)),
        snapshot_identity=incumbent.snapshot.identity,
        shots_by_probe=tuple((character * 64, 4096) for character in "1234"),
        observed_row_total_variations=(0.1, 0.2),
        fitted_row_total_variations=(0.1, 0.2),
        row_finite_shot_radii=(0.02, 0.02),
        residual_root_mean_square=0.0,
        maximum_absolute_residual=0.0,
        goodness_of_fit=1.0,
        confidence_level=0.95,
    )


def test_fit_candidate_builds_a_later_bound_model_without_mutation() -> None:
    incumbent = _incumbent()
    estimate = _estimate(incumbent)
    incumbent_payload = incumbent.noise_model.to_dict()

    candidate = fq.twin.fit_candidate(
        incumbent,
        estimate,
        captured_at="2026-08-15T10:30:00+08:00",
    )

    assert isinstance(candidate, fq.twin.TwinFittedCandidate)
    assert isinstance(candidate, fq.twin.QPUDigitalTwin)
    assert candidate.snapshot.identity != incumbent.snapshot.identity
    assert candidate.snapshot.captured_at == "2026-08-15T10:30:00+08:00"
    assert candidate.fit.parameter_estimate_identity == estimate.identity
    assert candidate.fit.probe_bank_identity == estimate.probe_bank_identity
    assert tuple(name for name, *_ in candidate.fit.fitted_parameters) == tuple(
        name for name, _ in estimate.parameter_values
    )
    assert candidate.fit.unidentified_parameters == ()
    assert "t1[Q20]" in candidate.fit.retained_parameters
    assert incumbent.noise_model.to_dict() == incumbent_payload
    assert incumbent.snapshot.captured_at == "2026-08-14T10:30:00+08:00"

    prediction = candidate.predict(fq.Circuit(2).h(0).cx(0, 1))
    assert prediction.snapshot_identity == candidate.snapshot.identity
    assert prediction.total_variation_from_ideal > 0.0
    assert candidate.noise_model.readout_rules[0].error.probabilities == (
        (0.99, 0.01),
        (0.01, 0.99),
    )
    channel_probabilities = tuple(
        dict(rule.channel.parameters)["probability"]
        for rule in candidate.noise_model.rules
    )
    assert channel_probabilities == pytest.approx((0.045, 0.06, 0.0625))


def test_fitted_candidate_round_trip_preserves_identity_and_predictions(
    tmp_path,
) -> None:
    incumbent = _incumbent()
    candidate = fq.twin.fit_candidate(
        incumbent,
        _estimate(incumbent),
        captured_at="2026-08-15T10:30:00+08:00",
    )
    destination = tmp_path / "fitted-candidate.json"
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    fq.twin.dump_twin(candidate, destination)
    fq.twin.dump_twin(candidate, destination)
    restored = fq.twin.load_twin(destination)

    assert isinstance(restored, fq.twin.TwinFittedCandidate)
    assert restored.snapshot.identity == candidate.snapshot.identity
    assert restored.fit.identity == candidate.fit.identity
    assert restored.predict(circuit) == candidate.predict(circuit)
    assert destination.stat().st_mode & 0o777 == 0o600


def test_fitted_candidate_load_rejects_tampered_provenance(tmp_path) -> None:
    incumbent = _incumbent()
    candidate = fq.twin.fit_candidate(
        incumbent,
        _estimate(incumbent),
        captured_at="2026-08-15T10:30:00+08:00",
    )
    destination = tmp_path / "tampered-candidate.json"
    fq.twin.dump_twin(candidate, destination)
    payload = json.loads(destination.read_text())
    payload["candidate_fit"]["fitted_parameters"][0]["value"] = 0.011
    destination.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Invalid QPU digital Twin"):
        fq.twin.load_twin(destination)


def test_existing_candidate_suite_accepts_the_fitted_twin() -> None:
    incumbent = _incumbent()
    candidate = fq.twin.fit_candidate(
        incumbent,
        _estimate(incumbent),
        captured_at="2026-08-15T10:30:00+08:00",
    )
    circuits = (
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).x(1).cx(1, 0),
    )

    suite = fq.twin.prepare_candidate_suite(
        incumbent,
        candidate,
        circuits,
        name="bounded-fit-holdout",
        shots=1024,
    )

    assert suite.circuit_count == 2
    assert all(
        trial.candidate_snapshot.identity == candidate.snapshot.identity
        for trial in suite.trials
    )


def test_candidate_fit_is_frozen_and_has_no_execution_authority() -> None:
    incumbent = _incumbent()
    candidate = fq.twin.fit_candidate(
        incumbent,
        _estimate(incumbent),
        captured_at="2026-08-15T10:30:00+08:00",
    )

    with pytest.raises(FrozenInstanceError):
        candidate.fit.parameter_family = "changed"  # type: ignore[misc]
    for name in ("submit", "poll", "retry", "promote", "release", "route"):
        assert not hasattr(candidate.fit, name)


def test_candidate_fit_fails_closed_on_changed_binding_or_time() -> None:
    incumbent = _incumbent()
    estimate = _estimate(incumbent)

    with pytest.raises(ValueError, match="target"):
        fq.twin.fit_candidate(
            incumbent,
            replace(estimate, target="quafu:Baihua"),
            captured_at="2026-08-15T10:30:00+08:00",
        )
    with pytest.raises(ValueError, match="mapping"):
        fq.twin.fit_candidate(
            incumbent,
            replace(estimate, physical_qubits=(27, 20)),
            captured_at="2026-08-15T10:30:00+08:00",
        )
    with pytest.raises(ValueError, match="incumbent snapshot"):
        fq.twin.fit_candidate(
            incumbent,
            replace(estimate, snapshot_identity="a" * 64),
            captured_at="2026-08-15T10:30:00+08:00",
        )
    with pytest.raises(ValueError, match="later"):
        fq.twin.fit_candidate(
            incumbent,
            estimate,
            captured_at=incumbent.snapshot.captured_at,
        )


def test_candidate_fit_rejects_unrepresentable_local_burden() -> None:
    incumbent = _incumbent()
    estimate = _estimate(incumbent)
    changed_values = tuple(
        (name, 0.8 if name == "single_qubit_pauli[Q20]" else value)
        for name, value in estimate.parameter_values
    )
    changed_intervals = tuple((name, value, value) for name, value in changed_values)

    with pytest.raises(ValueError, match="representable depolarizing range"):
        fq.twin.fit_candidate(
            incumbent,
            replace(
                estimate,
                parameter_values=changed_values,
                confidence_intervals=changed_intervals,
            ),
            captured_at="2026-08-15T10:30:00+08:00",
        )


def test_candidate_fit_rejects_a_coupler_outside_the_mapping() -> None:
    incumbent = _incumbent()
    estimate = _estimate(incumbent)
    changed_values = (
        *estimate.parameter_values,
        ("two_qubit_pauli[Q20->Q99]", 0.01),
    )
    changed_intervals = (
        *estimate.confidence_intervals,
        ("two_qubit_pauli[Q20->Q99]", 0.0, 0.02),
    )

    with pytest.raises(ValueError, match="incumbent mapping"):
        fq.twin.fit_candidate(
            incumbent,
            replace(
                estimate,
                parameter_values=changed_values,
                confidence_intervals=changed_intervals,
            ),
            captured_at="2026-08-15T10:30:00+08:00",
        )
