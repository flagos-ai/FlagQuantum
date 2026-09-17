"""Repeated-validation scenarios for QPU digital twins."""

from __future__ import annotations

import hashlib
import math

import pytest

import flagquantum as fq
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)
from flagquantum.twin import TwinExperiment, TwinValidationSeries

pytestmark = pytest.mark.integration


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


def _experiment():
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(), target="quafu:Baihua", qubits=(3, 4)
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    return (
        TwinExperiment.prepare(
            twin,
            circuit,
            name="repeated-bell",
            shots=1024,
        ),
        circuit,
    )


def _report(experiment, *, task_id, counts):
    digest = hashlib.sha256(experiment.submitted_qasm.encode()).hexdigest()
    handle = ProviderTaskHandle(
        provider="quafu",
        task_id=task_id,
        backend_name=experiment.backend_name,
        payload={
            "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": digest,
            "deployment_artifact_sha256": digest,
            "submitted_qasm_sha256": digest,
            "compiler": None,
            "target_qubits": list(experiment.target_qubits),
        },
    )
    result = DeploymentResult(
        handle=handle,
        counts=counts,
        shots=sum(counts.values()),
        metadata=build_result_metadata(
            handle,
            {"transpiled": experiment.submitted_qasm},
        ),
    )
    return experiment.validate_result(result, receipt=handle)


def test_validation_series_separates_accuracy_repeatability_and_shot_radius():
    experiment, circuit = _experiment()
    first = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )
    second = _report(
        experiment,
        task_id="task-2",
        counts={"00": 480, "01": 32, "10": 32, "11": 480},
    )

    series = experiment.validation_series([first, second], circuit=circuit)

    twin_distances = (
        first.validation.twin_hardware_total_variation,
        second.validation.twin_hardware_total_variation,
    )
    ideal_distances = (
        first.validation.ideal_hardware_total_variation,
        second.validation.ideal_hardware_total_variation,
    )
    simultaneous_confidence = 1.0 - (1.0 - 0.95) / 2.0
    radius = math.sqrt(
        (math.log(14.0) - math.log1p(-simultaneous_confidence)) / (2.0 * 1024.0)
    )
    assert series.repetitions == 2
    assert series.total_shots == 2048
    assert series.mean_twin_hardware_total_variation == pytest.approx(
        sum(twin_distances) / 2.0
    )
    assert series.maximum_twin_hardware_total_variation == pytest.approx(
        max(twin_distances)
    )
    assert series.mean_ideal_hardware_total_variation == pytest.approx(
        sum(ideal_distances) / 2.0
    )
    assert series.mean_hardware_repeatability_total_variation == pytest.approx(0.0625)
    assert series.maximum_hardware_repeatability_total_variation == pytest.approx(
        0.0625
    )
    assert series.simultaneous_finite_shot_tv_radius == pytest.approx(radius)
    assert series.verified_tv_error_bound == pytest.approx(
        min(1.0, max(twin_distances) + radius)
    )
    assert series.mean_twin_qpu_agreement == pytest.approx(
        1.0 - sum(twin_distances) / 2.0
    )
    assert series.mean_ideal_qpu_agreement == pytest.approx(
        1.0 - sum(ideal_distances) / 2.0
    )
    assert series.mean_qpu_repeatability == pytest.approx(0.9375)


def test_validation_series_produces_exact_circuit_evidence(tmp_path):
    experiment, circuit = _experiment()
    report = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )

    series = experiment.validation_series((report,), circuit=circuit)
    evidence = series.to_evidence()
    single_evidence = experiment.evidence_from_report(report, circuit=circuit)
    destination = tmp_path / "series-evidence.json"
    fq.twin.dump_evidence(evidence, destination)
    restored = fq.twin.load_evidence(destination)

    assert isinstance(series, TwinValidationSeries)
    assert series.mean_hardware_repeatability_total_variation is None
    assert series.maximum_hardware_repeatability_total_variation is None
    assert series.mean_qpu_repeatability is None
    assert evidence.evidence_identity == series.identity
    assert evidence.verified_tv_error_bound == series.verified_tv_error_bound
    assert evidence.verified_tv_error_bound == pytest.approx(
        single_evidence.verified_tv_error_bound
    )
    assert evidence.verified_circuit_identities == (circuit.to_ir().content_hash,)
    assert restored == evidence


def test_validation_series_persistence_is_canonical_and_non_replacing(tmp_path):
    experiment, circuit = _experiment()
    first = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )
    second = _report(
        experiment,
        task_id="task-2",
        counts={"00": 480, "01": 32, "10": 32, "11": 480},
    )
    series = experiment.validation_series((first, second), circuit=circuit)
    destination = tmp_path / "twin-validation.json"

    fq.twin.dump_validation_series(series, destination)
    original = destination.read_bytes()
    restored = fq.twin.load_validation_series(destination)
    fq.twin.dump_validation_series(series, destination)

    assert restored == series
    assert restored.identity == series.identity
    assert destination.read_bytes() == original

    different = experiment.validation_series((first,), circuit=circuit)
    with pytest.raises(ValueError, match="Refusing to replace different"):
        fq.twin.dump_validation_series(different, destination)


def test_validation_series_loader_rejects_noncanonical_payloads(tmp_path):
    experiment, circuit = _experiment()
    report = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )
    payload = experiment.validation_series((report,), circuit=circuit).to_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="fields do not match"):
        TwinValidationSeries.from_dict(payload)

    del payload["unexpected"]
    del payload["schema"]
    with pytest.raises(ValueError, match="fields do not match"):
        TwinValidationSeries.from_dict(payload)

    destination = tmp_path / "invalid.json"
    destination.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        fq.twin.load_validation_series(destination)


def test_validation_series_writer_preserves_invalid_existing_file(tmp_path):
    experiment, circuit = _experiment()
    report = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )
    series = experiment.validation_series((report,), circuit=circuit)
    destination = tmp_path / "invalid.json"
    destination.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to replace invalid"):
        fq.twin.dump_validation_series(series, destination)

    assert destination.read_text(encoding="utf-8") == "not-json\n"


def test_validation_series_rejects_duplicate_hardware_tasks():
    experiment, circuit = _experiment()
    report = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )

    with pytest.raises(ValueError, match="distinct hardware tasks"):
        experiment.validation_series((report, report), circuit=circuit)


def test_validation_series_reuses_fail_closed_report_binding():
    experiment, circuit = _experiment()
    report = _report(
        experiment,
        task_id="task-1",
        counts={"00": 512, "11": 512},
    )

    with pytest.raises(RuntimeError, match="circuit does not match"):
        experiment.validation_series(
            (report,),
            circuit=fq.Circuit(2).x(0),
        )


@pytest.mark.parametrize("reports", [(), [], None])
def test_validation_series_requires_reports(reports):
    experiment, circuit = _experiment()

    with pytest.raises((TypeError, ValueError)):
        experiment.validation_series(reports, circuit=circuit)
