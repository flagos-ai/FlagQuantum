"""Executable contract for the candidate ``fq.twin`` v1 surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_twin_v1_contract_is_formally_frozen() -> None:
    contract = json.loads((ROOT / "contracts" / "twin-v1-candidate.json").read_text())

    assert contract["status"] == "frozen"
    assert contract["candidate_is_frozen_contract"] is True
    assert contract["public_schema_defaults"]["TwinEvidenceEnvelope"] == (
        "flagquantum.twin_evidence_envelope.v1"
    )
    assert contract["public_schema_defaults"]["TwinValidationSeries"] == (
        "flagquantum.twin_validation_series.v1"
    )
    assert contract["public_schema_defaults"]["TwinCircuitSupport"] == (
        "flagquantum.twin_circuit_support.v1"
    )
    assert contract["public_schema_defaults"]["TwinConnectedRegion"] == (
        "flagquantum.twin_connected_region.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionCoverage"] == (
        "flagquantum.twin_region_coverage.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionModel"] == (
        "flagquantum.twin_region_model.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionValidationSuite"] == (
        "flagquantum.twin_region_validation_suite.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionSuiteEvaluation"] == (
        "flagquantum.twin_region_suite_evaluation.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionValidationHistory"] == (
        "flagquantum.twin_region_validation_history.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionHoldoutStudy"] == (
        "flagquantum.twin_region_holdout_study.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionHoldoutEvaluation"] == (
        "flagquantum.twin_region_holdout_evaluation.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionHoldoutHistory"] == (
        "flagquantum.twin_region_holdout_history.v1"
    )
    assert contract["public_schema_defaults"]["TwinRegionHoldoutEvolution"] == (
        "flagquantum.twin_region_holdout_evolution.v1"
    )
    assert contract["public_literal_values"]["TwinRegionCoverageStatus"] == [
        "covered",
        "out_of_scope",
    ]


def _device_noise_model() -> NoiseModel:
    profile = DeviceNoiseProfile(
        qubits=(
            QubitNoiseCalibration(0, t1=40_000.0, t2=60_000.0),
            QubitNoiseCalibration(1, t1=35_000.0, t2=50_000.0),
        ),
        gate_durations=(GateDuration("h", 64.0), GateDuration("cx", 224.0)),
        source="contract-fixture",
        captured_at="2026-08-14T10:30:00+08:00",
    )
    return NoiseModel.from_device_profile(profile)


def test_provider_neutral_twin_golden_path_is_offline_and_deterministic() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    first = fq.twin.from_noise_model(
        _device_noise_model(),
        target="acme:research-qpu",
        qubits=(12, 13),
    )
    second = fq.twin.from_noise_model(
        _device_noise_model(),
        target="acme:research-qpu",
        qubits=(12, 13),
    )
    prediction = first.predict(circuit)

    assert first.snapshot == second.snapshot
    assert first.snapshot.provider == "acme"
    assert first.snapshot.backend_name == "research-qpu"
    assert first.snapshot.physical_qubits == (12, 13)
    assert prediction.snapshot_identity == first.snapshot.identity
    assert prediction.circuit_identity == circuit.to_ir().content_hash
    assert sum(prediction.twin_probabilities) == pytest.approx(1.0)


def test_prepare_freezes_identity_without_submitting_hardware() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    twin = fq.twin.from_noise_model(
        _device_noise_model(),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )

    experiment = fq.twin.TwinExperiment.prepare(
        twin,
        circuit,
        name="frozen-bell",
        shots=1024,
    )

    assert experiment.backend_name == "Shenglian"
    assert experiment.target_qubits == (20, 27)
    assert experiment.snapshot_identity == twin.snapshot.identity
    assert experiment.submitted_qasm.startswith("OPENQASM 2.0;")
    assert "measure q[0] -> c[0];" in experiment.submitted_qasm
    assert "measure q[1] -> c[1];" in experiment.submitted_qasm


def test_class_construction_shortcuts_do_not_compete_with_fq_factories() -> None:
    assert not hasattr(fq.twin.QPUDigitalTwin, "from_noise_model")
    assert not hasattr(fq.twin.QPUDigitalTwin, "from_quafu_chip_info")
    assert not hasattr(fq.twin.QPUDigitalTwin, "from_quafu_provider")
