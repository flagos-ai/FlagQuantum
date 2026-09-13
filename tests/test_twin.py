"""Scenario tests for the first QPU digital-twin vertical slice."""

from __future__ import annotations

import copy

import pytest

import flagquantum as fq
from flagquantum.twin import (
    QPUDigitalTwin,
    TwinEvidenceEnvelope,
    TwinEvidenceReport,
    TwinExperiment,
    TwinHardwareReport,
    TwinPrediction,
    TwinSnapshot,
    TwinValidationReport,
)


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q3": {"T1": 40.0, "T2": 60.0, "fidelity": 1.0, "length": 6.4e-8},
            "Q4": {"T1": 35.0, "T2": 50.0, "fidelity": 1.0, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [3, 4],
                "fidelity": 1.0,
                "length": 2.24e-7,
            }
        },
    }


def test_twin_namespace_is_small_and_domain_named():
    import flagquantum.twin as fqt

    assert fqt.__all__ == (
        "QPUDigitalTwin",
        "TwinEvidenceEnvelope",
        "TwinEvidenceReport",
        "TwinEvidenceStatus",
        "TwinExperiment",
        "TwinHardwareReport",
        "TwinPrediction",
        "TwinSnapshot",
        "TwinValidationReport",
    )
    assert all(
        value in fqt.__all__
        for value in (
            QPUDigitalTwin.__name__,
            TwinEvidenceEnvelope.__name__,
            TwinEvidenceReport.__name__,
            TwinExperiment.__name__,
            TwinHardwareReport.__name__,
            TwinPrediction.__name__,
            TwinSnapshot.__name__,
            TwinValidationReport.__name__,
            "TwinEvidenceStatus",
        )
    )


def test_quafu_twin_freezes_calibration_and_predicts_decoherence():
    twin = QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(), backend_name="Baihua", physical_qubits=(3, 4)
    )

    prediction = twin.predict(fq.Circuit(2).h(0).cx(0, 1))

    assert twin.snapshot.captured_at == "2026-08-14 10:30:00+08:00"
    assert twin.snapshot.physical_qubits == (3, 4)
    assert (
        twin.snapshot.calibration_identity == twin.noise_model.device_profile.identity
    )
    assert twin.snapshot.noise_model_identity == twin.noise_model.identity
    assert prediction.snapshot_identity == twin.snapshot.identity
    assert sum(prediction.twin_probabilities) == pytest.approx(1.0, abs=1e-6)
    assert prediction.total_variation_from_ideal > 0


def test_twin_applies_readout_and_compares_hardware_counts():
    twin = QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(),
        backend_name="Baihua",
        physical_qubits=(3, 4),
        readout_confusion_matrices=(
            ((0.8, 0.2), (0.1, 0.9)),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
    )

    prediction = twin.predict(fq.Circuit(2))
    report = prediction.compare_counts({"00": 80, "10": 20})

    assert prediction.twin_probabilities == pytest.approx((0.8, 0.0, 0.2, 0.0))
    assert report.twin_hardware_total_variation == pytest.approx(0.0, abs=1e-8)
    assert report.ideal_hardware_total_variation == pytest.approx(0.2)
    assert report.total_variation_improvement == pytest.approx(0.2)
    assert report.outperforms_ideal_baseline is True
    assert report.snapshot_identity == prediction.snapshot_identity


def test_snapshot_identity_changes_with_selected_calibration():
    changed = copy.deepcopy(_chip_info())
    changed["qubits_info"]["Q3"]["T1"] = 39.0
    first = QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(), backend_name="Baihua", physical_qubits=(3, 4)
    )
    second = QPUDigitalTwin.from_quafu_chip_info(
        changed, backend_name="Baihua", physical_qubits=(3, 4)
    )

    assert first.snapshot.identity != second.snapshot.identity


def test_twin_can_freeze_calibration_fetched_by_provider():
    class Provider:
        provider = "quafu"

        def fetch_chip_info(self, backend_name):
            assert backend_name == "Baihua"
            return _chip_info()

    twin = QPUDigitalTwin.from_quafu_provider(
        Provider(), backend_name="Baihua", physical_qubits=(3, 4)
    )

    assert twin.snapshot.provider == "quafu"
    assert twin.snapshot.backend_name == "Baihua"


def test_twin_copies_and_guards_its_noise_model():
    from flagquantum.noise import bit_flip_channel
    from flagquantum.remote.qpu import quafu_noise_model_from_chip_info

    source = quafu_noise_model_from_chip_info(_chip_info(), physical_qubits=(3, 4))
    twin = QPUDigitalTwin.from_noise_model(
        source,
        provider="quafu",
        backend_name="Baihua",
        physical_qubits=(3, 4),
    )

    assert twin.noise_model is not source
    twin.noise_model.add("x", bit_flip_channel(0.1))
    with pytest.raises(RuntimeError, match="modified"):
        twin.predict(fq.Circuit(2))


def test_twin_rejects_mismatched_circuit_and_invalid_counts():
    twin = QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(), backend_name="Baihua", physical_qubits=(3, 4)
    )
    with pytest.raises(ValueError, match="width"):
        twin.predict(fq.Circuit(1))

    prediction = twin.predict(fq.Circuit(2))
    with pytest.raises(ValueError, match="fixed-width"):
        prediction.compare_counts({"0": 10})
