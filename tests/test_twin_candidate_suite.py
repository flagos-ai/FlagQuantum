from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import flagquantum as fq
from examples.remote import quafu_twin_candidate_suite
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)
from flagquantum.twin import TwinCandidateSubmission, TwinPrediction

pytestmark = pytest.mark.unit


def _chip_info(captured_at: str):
    return {
        "calibration_time": captured_at,
        "basis_gates": ["h", "x", "rx", "ry", "rz", "cz"],
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


def _prediction(base: TwinPrediction, probabilities: tuple[float, ...]):
    distance = 0.5 * sum(
        abs(ideal - predicted)
        for ideal, predicted in zip(base.ideal_probabilities, probabilities)
    )
    return replace(
        base,
        twin_probabilities=probabilities,
        total_variation_from_ideal=distance,
    )


def _suite():
    incumbent = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-14 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    candidate = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-15 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    circuits = (
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).x(0).cx(0, 1),
        fq.Circuit(2).h(1).cx(1, 0).x(1),
    )
    base = fq.twin.prepare_candidate_suite(
        incumbent,
        candidate,
        circuits,
        name="candidate-workloads",
        shots=1024,
    )
    trials = []
    for trial in base.trials:
        trials.append(
            replace(
                trial,
                incumbent_prediction=_prediction(
                    trial.incumbent_prediction,
                    (0.70, 0.0, 0.0, 0.30),
                ),
                experiment=replace(
                    trial.experiment,
                    prediction=_prediction(
                        trial.experiment.prediction,
                        (0.98, 0.0, 0.0, 0.02),
                    ),
                ),
            )
        )
    return fq.twin.TwinCandidateSuite(tuple(trials)), circuits


def _submission(trial, index: int):
    experiment = trial.experiment
    receipt = ProviderTaskHandle(
        provider="quafu",
        task_id=f"candidate-suite-task-{index}",
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
    return TwinCandidateSubmission.from_receipt(trial, receipt)


def _result(submission: TwinCandidateSubmission):
    receipt = submission.receipt
    experiment = submission.trial.experiment
    return DeploymentResult(
        handle=receipt,
        counts={"00": 1024},
        shots=1024,
        metadata=build_result_metadata(
            receipt,
            {
                "chip": experiment.backend_name,
                "transpiled": experiment.submitted_qasm,
            },
        ),
    )


def test_candidate_suite_compares_distinct_circuits_with_simultaneous_confidence():
    suite, circuits = _suite()
    submissions = tuple(
        _submission(trial, index) for index, trial in enumerate(suite.trials, start=1)
    )
    results = tuple(_result(submission) for submission in submissions)

    evaluation = suite.validate_results(
        submissions,
        results,
        circuits=circuits,
        confidence_level=0.95,
    )

    assert evaluation.decision == "improved"
    assert evaluation.circuit_count == 3
    assert evaluation.total_shots == 3072
    assert evaluation.mean_incumbent_qpu_agreement == pytest.approx(0.70)
    assert evaluation.mean_candidate_qpu_agreement == pytest.approx(0.98)
    assert evaluation.mean_candidate_improvement == pytest.approx(0.28)
    assert all(
        item.confidence_level == pytest.approx(1.0 - 0.05 / 3.0)
        for item in evaluation.evaluations
    )
    assert evaluation.to_dict()["circuit_count"] == 3


def test_candidate_suite_round_trip_is_private_create_once(tmp_path):
    suite, _ = _suite()
    destination = tmp_path / "candidate-suite.json"

    fq.twin.dump_candidate_suite(suite, destination)
    fq.twin.dump_candidate_suite(suite, destination)
    restored = fq.twin.load_candidate_suite(destination)

    assert restored == suite
    assert restored.identity == suite.identity
    assert restored.circuit_count == 3
    assert restored.total_shots == 3072
    assert destination.stat().st_mode & 0o777 == 0o600

    changed = replace(suite, trials=tuple(reversed(suite.trials)))
    with pytest.raises(ValueError, match="different Twin candidate suite"):
        fq.twin.dump_candidate_suite(changed, destination)


def test_candidate_suite_fails_closed_on_missing_or_mismatched_bindings():
    suite, circuits = _suite()
    submissions = tuple(
        _submission(trial, index) for index, trial in enumerate(suite.trials, start=1)
    )
    results = tuple(_result(submission) for submission in submissions)

    with pytest.raises(ValueError, match="match every suite trial"):
        suite.validate_results(
            submissions[:-1],
            results,
            circuits=circuits,
        )
    with pytest.raises(ValueError, match="does not match its suite trial"):
        suite.validate_results(
            tuple(reversed(submissions)),
            results,
            circuits=circuits,
        )
    duplicate_task = replace(
        submissions[1],
        submission=replace(
            submissions[1].submission,
            receipt=replace(
                submissions[1].receipt,
                task_id=submissions[0].receipt.task_id,
            ),
        ),
    )
    with pytest.raises(ValueError, match="distinct QPU tasks"):
        suite.validate_results(
            (submissions[0], duplicate_task, submissions[2]),
            (results[0], _result(duplicate_task), results[2]),
            circuits=circuits,
        )


def test_candidate_suite_rejects_one_or_duplicate_circuits():
    incumbent = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-14 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    candidate = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-15 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="at least two circuits"):
        fq.twin.prepare_candidate_suite(
            incumbent,
            candidate,
            (circuit,),
            name="too-small",
            shots=1024,
        )
    with pytest.raises(ValueError, match="distinct circuits"):
        fq.twin.prepare_candidate_suite(
            incumbent,
            candidate,
            (circuit, circuit),
            name="duplicates",
            shots=1024,
        )


def test_quafu_suite_example_checkpoints_each_task_before_evaluation(
    monkeypatch, tmp_path
):
    incumbent = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-14 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    incumbent_path = tmp_path / "twin-incumbent.json"
    suite_path = tmp_path / "candidate-suite.json"
    fq.twin.dump_twin(incumbent, incumbent_path)

    class Provider:
        provider = "quafu"

        def __init__(self):
            self.submissions = 0
            self.qasm_by_task: dict[str, str] = {}

        def fetch_chip_info(self, chip):
            assert chip == "Shenglian"
            return _chip_info("2026-08-15 10:30:00")

        def submit_qasm(self, qasm, *, chip, name, shots, target_qubits):
            self.submissions += 1
            task_id = f"suite-example-task-{self.submissions}"
            self.qasm_by_task[task_id] = qasm
            digest = hashlib.sha256(qasm.encode()).hexdigest()
            return ProviderTaskHandle(
                provider="quafu",
                task_id=task_id,
                backend_name=chip,
                payload={
                    "deployment_receipt_schema": ("flagquantum_submission_receipt_v1"),
                    "deployment_package_schema": "flagquantum_submitted_qasm_v1",
                    "deployment_program_format": "openqasm-2",
                    "routing_evidence_sha256": digest,
                    "deployment_artifact_sha256": digest,
                    "submitted_qasm_sha256": digest,
                    "compiler": None,
                    "target_qubits": list(target_qubits),
                },
            )

        def query_status(self, receipt):
            return "Finished"

        def fetch_result(self, receipt):
            return DeploymentResult(
                handle=receipt,
                counts={"00": 512, "11": 512},
                shots=1024,
                metadata=build_result_metadata(
                    receipt,
                    {
                        "chip": "Shenglian",
                        "transpiled": self.qasm_by_task[receipt.task_id],
                    },
                ),
            )

    provider = Provider()
    monkeypatch.setattr(quafu_twin_candidate_suite, "INCUMBENT_PATH", incumbent_path)
    monkeypatch.setattr(quafu_twin_candidate_suite, "SUITE_PATH", suite_path)
    monkeypatch.setattr(
        quafu_twin_candidate_suite,
        "SUBMISSION_PREFIX",
        str(tmp_path / "candidate-submission"),
    )

    quafu_twin_candidate_suite.prepare(provider)
    assert provider.submissions == 0
    for index in range(1, 4):
        quafu_twin_candidate_suite.submit_one(provider, index)
        assert provider.submissions == index

    quafu_twin_candidate_suite.evaluate(provider)
    assert provider.submissions == 3
