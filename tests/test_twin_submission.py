"""Persistence tests for resumable QPU Twin submissions."""

from __future__ import annotations

import hashlib
import json

import pytest

import flagquantum as fq
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {"T1": 40.0, "T2": 60.0, "fidelity": 1.0, "length": 6.4e-8},
            "Q27": {"T1": 35.0, "T2": 50.0, "fidelity": 1.0, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 1.0,
                "length": 2.24e-7,
            }
        },
    }


def _experiment():
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(), target="quafu:Shenglian", qubits=(20, 27)
    )
    return fq.twin.TwinExperiment.prepare(
        twin,
        circuit,
        name="resumable-bell",
        shots=1024,
    )


def _receipt(experiment, *, task_id="task-42"):
    digest = experiment.submitted_qasm_identity
    return ProviderTaskHandle(
        provider="quafu",
        task_id=task_id,
        backend_name="Shenglian",
        payload={
            "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": digest,
            "deployment_artifact_sha256": digest,
            "submitted_qasm_sha256": digest,
            "compiler": None,
            "target_qubits": [20, 27],
            "submit": {"provider_internal": "not persisted"},
        },
    )


def test_submission_round_trip_resumes_validation_without_submission(tmp_path):
    experiment = _experiment()
    submission = fq.twin.TwinSubmission.from_receipt(experiment, _receipt(experiment))
    destination = tmp_path / "twin-submission.json"

    fq.twin.dump_submission(submission, destination)
    restored = fq.twin.load_submission(destination)

    assert restored.identity == submission.identity
    assert restored.experiment == experiment
    assert restored.receipt.task_id == "task-42"
    assert "submit" not in restored.receipt.payload
    assert "provider_internal" not in destination.read_text(encoding="utf-8")
    assert destination.stat().st_mode & 0o777 == 0o600

    result = DeploymentResult(
        handle=restored.receipt,
        counts={"00": 512, "11": 512},
        shots=1024,
        metadata=build_result_metadata(
            restored.receipt,
            {
                "chip": "Shenglian",
                "transpiled": experiment.submitted_qasm,
            },
        ),
    )
    report = restored.validate_result(result)

    assert report.task_id == "task-42"
    assert report.validation.shots == 1024


def test_submission_copy_is_immutable_and_save_is_idempotent(tmp_path):
    experiment = _experiment()
    receipt = _receipt(experiment)
    submission = fq.twin.TwinSubmission.from_receipt(experiment, receipt)
    destination = tmp_path / "twin-submission.json"

    receipt.payload["target_qubits"].append(34)
    assert tuple(submission.receipt.payload["target_qubits"]) == (20, 27)
    with pytest.raises(TypeError):
        submission.receipt.payload["task"] = "changed"

    fq.twin.dump_submission(submission, destination)
    fq.twin.dump_submission(submission, destination)


def test_submission_rejects_mismatched_receipt_and_unknown_fields(tmp_path):
    experiment = _experiment()
    wrong_digest = hashlib.sha256(b"different").hexdigest()
    receipt = _receipt(experiment)
    receipt.payload["submitted_qasm_sha256"] = wrong_digest

    with pytest.raises(ValueError, match="frozen QASM"):
        fq.twin.TwinSubmission.from_receipt(experiment, receipt)

    submission = fq.twin.TwinSubmission.from_receipt(experiment, _receipt(experiment))
    payload = submission.to_dict()
    payload["unexpected"] = True
    destination = tmp_path / "invalid.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        fq.twin.load_submission(destination)


def test_submission_refuses_to_replace_other_or_invalid_content(tmp_path):
    experiment = _experiment()
    first = fq.twin.TwinSubmission.from_receipt(experiment, _receipt(experiment))
    second = fq.twin.TwinSubmission.from_receipt(
        experiment, _receipt(experiment, task_id="task-43")
    )
    destination = tmp_path / "twin-submission.json"

    fq.twin.dump_submission(first, destination)
    with pytest.raises(ValueError, match="different Twin submission"):
        fq.twin.dump_submission(second, destination)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Twin submission"):
        fq.twin.dump_submission(first, invalid)
