"""Prospective connected-region holdout validation scenarios."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import flagquantum as fq
from examples.remote import quafu_twin_region_holdout
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)

pytestmark = pytest.mark.integration


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {"T1": 41.0, "T2": 61.0, "fidelity": 0.997, "length": 6.4e-8},
            "Q27": {"T1": 40.0, "T2": 60.0, "fidelity": 0.998, "length": 6.4e-8},
            "Q34": {"T1": 42.0, "T2": 62.0, "fidelity": 0.996, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 0.985,
                "length": 2.24e-7,
            },
            "C1": {
                "qubits_index": [27, 34],
                "fidelity": 0.984,
                "length": 2.24e-7,
            },
        },
    }


def _cell(qubits: tuple[int, int]):
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(),
        target="quafu:Shenglian",
        qubits=qubits,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=qubits,
        supported_operations=("h", "cx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(circuit.to_ir().content_hash,),
        evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    return twin, fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=5,
    )


def _region_twin() -> fq.twin.TwinRegionModel:
    return fq.twin.compose_region_twin((_cell((20, 27)), _cell((27, 34))))


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


def _study() -> fq.twin.TwinRegionHoldoutStudy:
    return fq.twin.prepare_region_holdout_study(
        _region_twin(),
        _reference_circuits(),
        _holdout_circuits(),
        physical_qubits=(20, 27, 34),
        name="regional-holdout",
        shots=1024,
        repetitions=2,
    )


def _submission(
    experiment: fq.twin.TwinExperiment,
    task_id: str,
) -> fq.twin.TwinSubmission:
    receipt = ProviderTaskHandle(
        provider="quafu",
        task_id=task_id,
        backend_name=experiment.backend_name,
        payload={
            "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": experiment.submitted_qasm_identity,
            "deployment_artifact_sha256": experiment.submitted_qasm_identity,
            "submitted_qasm_sha256": experiment.submitted_qasm_identity,
            "compiler": None,
            "target_qubits": list(experiment.target_qubits),
        },
    )
    return fq.twin.TwinSubmission.from_receipt(experiment, receipt)


def _suite_inputs(
    suite: fq.twin.TwinRegionValidationSuite,
    *,
    prefix: str,
    offset_step: int,
):
    submissions = []
    results = []
    task_index = 0
    for experiment in suite.experiments:
        for repetition in range(suite.repetitions):
            task_index += 1
            submission = _submission(experiment, f"{prefix}-{task_index}")
            offset = repetition * offset_step
            result = DeploymentResult(
                handle=submission.receipt,
                counts={"000": 512 - offset, "111": 512 + offset},
                shots=1024,
                metadata=build_result_metadata(
                    submission.receipt,
                    {"transpiled": submission.experiment.submitted_qasm},
                ),
            )
            submissions.append(submission)
            results.append(result)
    return tuple(submissions), tuple(results)


def test_holdout_study_freezes_disjoint_predictions_without_submission() -> None:
    study = _study()

    assert study.reference_circuit_count == 2
    assert study.holdout_circuit_count == 2
    assert study.planned_task_count == 8
    assert study.planned_shots == 8192
    assert set(
        item.prediction.circuit_identity for item in study.reference_suite.experiments
    ).isdisjoint(
        item.prediction.circuit_identity for item in study.holdout_suite.experiments
    )


def test_holdout_study_validates_both_groups_under_one_confidence() -> None:
    study = _study()
    reference_submissions, reference_results = _suite_inputs(
        study.reference_suite,
        prefix="reference",
        offset_step=8,
    )
    holdout_submissions, holdout_results = _suite_inputs(
        study.holdout_suite,
        prefix="holdout",
        offset_step=16,
    )

    evaluation = study.validate_results(
        reference_submissions,
        reference_results,
        holdout_submissions,
        holdout_results,
        reference_circuits=_reference_circuits(),
        holdout_circuits=_holdout_circuits(),
        confidence_level=0.95,
    )

    assert evaluation.reference_circuit_count == 2
    assert evaluation.holdout_circuit_count == 2
    assert evaluation.task_count == 8
    assert evaluation.total_shots == 8192
    assert evaluation.confidence_level == pytest.approx(0.95)
    assert evaluation.reference_evaluation.confidence_level == pytest.approx(0.975)
    assert evaluation.holdout_evaluation.confidence_level == pytest.approx(0.975)
    assert evaluation.holdout_twin_qpu_tv_increase == pytest.approx(
        evaluation.reference_twin_qpu_agreement - evaluation.holdout_twin_qpu_agreement
    )
    assert evaluation.holdout_simultaneous_tv_error_bound == (
        evaluation.holdout_evaluation.simultaneous_tv_error_bound
    )


def test_holdout_study_rejects_circuit_leakage() -> None:
    reference = _reference_circuits()

    with pytest.raises(ValueError, match="must be disjoint"):
        fq.twin.prepare_region_holdout_study(
            _region_twin(),
            reference,
            (reference[0], _holdout_circuits()[0]),
            physical_qubits=(20, 27, 34),
            name="leaked-holdout",
            shots=1024,
        )


def test_holdout_study_rejects_cross_group_task_reuse() -> None:
    study = _study()
    reference_submissions, reference_results = _suite_inputs(
        study.reference_suite,
        prefix="reference",
        offset_step=8,
    )
    holdout_submissions, holdout_results = _suite_inputs(
        study.holdout_suite,
        prefix="holdout",
        offset_step=16,
    )
    reused = replace(
        holdout_submissions[0],
        receipt=replace(
            holdout_submissions[0].receipt,
            task_id=reference_submissions[0].receipt.task_id,
        ),
    )

    with pytest.raises(ValueError, match="distinct QPU tasks"):
        study.validate_results(
            reference_submissions,
            reference_results,
            (reused, *holdout_submissions[1:]),
            holdout_results,
            reference_circuits=_reference_circuits(),
            holdout_circuits=_holdout_circuits(),
        )


def test_holdout_study_round_trip_is_private_and_create_once(tmp_path) -> None:
    study = _study()
    destination = tmp_path / "holdout-study.json"

    fq.twin.dump_region_holdout_study(study, destination)
    original = destination.read_bytes()
    restored = fq.twin.load_region_holdout_study(destination)
    fq.twin.dump_region_holdout_study(study, destination)

    assert restored == study
    assert destination.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o600


def test_holdout_study_rejects_changed_repetition_count() -> None:
    study = _study()
    changed = replace(
        study.holdout_suite,
        repetitions=3,
    )

    with pytest.raises(ValueError, match="one regional Twin, mapping"):
        fq.twin.TwinRegionHoldoutStudy(
            reference_suite=study.reference_suite,
            holdout_suite=changed,
        )


def test_complete_holdout_example_is_checkpointed_and_executable(
    monkeypatch,
    tmp_path,
) -> None:
    twin_a, support_a = _cell((20, 27))
    twin_b, support_b = _cell((27, 34))
    paths = {
        "CELL_A_TWIN": tmp_path / "cell-a-twin.json",
        "CELL_A_SUPPORT": tmp_path / "cell-a-support.json",
        "CELL_B_TWIN": tmp_path / "cell-b-twin.json",
        "CELL_B_SUPPORT": tmp_path / "cell-b-support.json",
        "STUDY_PATH": tmp_path / "holdout-study.json",
        "REFERENCE_SUPPORT_PATH": tmp_path / "reference-support.json",
        "HOLDOUT_SUPPORT_PATH": tmp_path / "holdout-support.json",
    }
    fq.twin.dump_twin(twin_a, paths["CELL_A_TWIN"])
    fq.twin.dump_circuit_support(support_a, paths["CELL_A_SUPPORT"])
    fq.twin.dump_twin(twin_b, paths["CELL_B_TWIN"])
    fq.twin.dump_circuit_support(support_b, paths["CELL_B_SUPPORT"])
    for name, path in paths.items():
        monkeypatch.setattr(quafu_twin_region_holdout, name, path)
    monkeypatch.setattr(
        quafu_twin_region_holdout,
        "SUBMISSION_PREFIX",
        str(tmp_path / "submission"),
    )

    class Provider:
        provider = "quafu"

        def __init__(self) -> None:
            self.submissions = 0
            self.programs: dict[str, str] = {}

        def submit_qasm(self, qasm, *, chip, name, shots, target_qubits):
            assert chip == "Shenglian"
            assert shots == 1024
            assert target_qubits == (20, 27, 34)
            self.submissions += 1
            task_id = f"holdout-example-{self.submissions}"
            self.programs[task_id] = qasm
            digest = hashlib.sha256(qasm.encode()).hexdigest()
            return ProviderTaskHandle(
                provider="quafu",
                task_id=task_id,
                backend_name=chip,
                payload={
                    "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
                    "deployment_package_schema": "flagquantum_submitted_qasm_v1",
                    "deployment_program_format": "openqasm-2",
                    "routing_evidence_sha256": digest,
                    "deployment_artifact_sha256": digest,
                    "submitted_qasm_sha256": digest,
                    "compiler": None,
                    "target_qubits": list(target_qubits),
                },
            )

        def fetch_result(self, receipt):
            return DeploymentResult(
                handle=receipt,
                counts={"000": 512, "111": 512},
                shots=1024,
                metadata=build_result_metadata(
                    receipt,
                    {"transpiled": self.programs[receipt.task_id]},
                ),
            )

    provider = Provider()
    quafu_twin_region_holdout.prepare()
    study = fq.twin.load_region_holdout_study(paths["STUDY_PATH"])
    assert study.planned_task_count == 8
    assert provider.submissions == 0

    for index in range(1, study.planned_task_count + 1):
        quafu_twin_region_holdout.submit_one(provider, index)
    with pytest.raises(FileExistsError, match="refusing to resubmit"):
        quafu_twin_region_holdout.submit_one(provider, 1)
    assert provider.submissions == 8

    quafu_twin_region_holdout.evaluate(provider)

    reference = fq.twin.load_circuit_support(paths["REFERENCE_SUPPORT_PATH"])
    holdout = fq.twin.load_circuit_support(paths["HOLDOUT_SUPPORT_PATH"])
    assert len(reference.evidence.verified_circuit_identities) == 2
    assert len(holdout.evidence.verified_circuit_identities) == 2
