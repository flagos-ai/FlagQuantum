"""Prospective incumbent-versus-candidate Twin comparisons."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import flagquantum as fq
from examples.remote import quafu_twin_candidate
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)
from flagquantum.twin import (
    TwinCandidateTrial,
    TwinPrediction,
    prepare_candidate_trial,
)

pytestmark = pytest.mark.unit


def _chip_info(captured_at: str):
    return {
        "calibration_time": captured_at,
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


def _prediction(base: TwinPrediction, probabilities: tuple[float, ...]):
    distance = 0.5 * sum(
        abs(ideal - predicted)
        for ideal, predicted in zip(base.ideal_probabilities, probabilities)
    )
    return TwinPrediction(
        snapshot_identity=base.snapshot_identity,
        circuit_identity=base.circuit_identity,
        n_wires=base.n_wires,
        ideal_probabilities=base.ideal_probabilities,
        twin_probabilities=probabilities,
        total_variation_from_ideal=distance,
    )


def _trial(
    incumbent_probabilities: tuple[float, ...],
    candidate_probabilities: tuple[float, ...],
):
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
    base = prepare_candidate_trial(
        incumbent,
        candidate,
        circuit,
        name="candidate-bell",
        shots=1024,
    )
    incumbent_prediction = _prediction(
        base.incumbent_prediction, incumbent_probabilities
    )
    candidate_prediction = _prediction(
        base.experiment.prediction, candidate_probabilities
    )
    experiment = replace(base.experiment, prediction=candidate_prediction)
    return (
        TwinCandidateTrial(
            incumbent_snapshot=base.incumbent_snapshot,
            candidate_snapshot=base.candidate_snapshot,
            incumbent_prediction=incumbent_prediction,
            experiment=experiment,
        ),
        circuit,
    )


def _receipt(trial: TwinCandidateTrial):
    experiment = trial.experiment
    return ProviderTaskHandle(
        provider="quafu",
        task_id="candidate-task-42",
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


def _result(trial: TwinCandidateTrial, counts: dict[str, int]):
    receipt = _receipt(trial)
    return receipt, DeploymentResult(
        handle=receipt,
        counts=counts,
        shots=sum(counts.values()),
        metadata=build_result_metadata(
            receipt,
            {
                "chip": trial.experiment.backend_name,
                "transpiled": trial.experiment.submitted_qasm,
            },
        ),
    )


def test_preparing_candidate_trial_is_offline_and_submission_is_explicit():
    trial, _ = _trial((0.7, 0.0, 0.0, 0.3), (0.98, 0.0, 0.0, 0.02))

    class Provider:
        provider = "quafu"

        def __init__(self):
            self.submissions = 0

        def submit_qasm(self, qasm, *, chip, name, shots, target_qubits):
            self.submissions += 1
            assert hashlib.sha256(qasm.encode()).hexdigest() == (
                trial.experiment.submitted_qasm_identity
            )
            assert (chip, name, shots, target_qubits) == (
                "Shenglian",
                "candidate-bell",
                1024,
                (20, 27),
            )
            return _receipt(trial)

    provider = Provider()
    assert provider.submissions == 0
    trial.experiment.submit(provider)
    assert provider.submissions == 1


@pytest.mark.parametrize(
    ("incumbent", "candidate", "decision"),
    (
        ((0.7, 0.0, 0.0, 0.3), (0.98, 0.0, 0.0, 0.02), "improved"),
        ((0.98, 0.0, 0.0, 0.02), (0.7, 0.0, 0.0, 0.3), "degraded"),
        ((0.90, 0.0, 0.0, 0.10), (0.95, 0.0, 0.0, 0.05), "inconclusive"),
    ),
)
def test_candidate_decision_uses_one_shared_hardware_result(
    incumbent, candidate, decision
):
    trial, circuit = _trial(incumbent, candidate)
    receipt, result = _result(trial, {"00": 1024})

    evaluation = trial.validate_result(
        result,
        receipt=receipt,
        circuit=circuit,
        confidence_level=0.95,
    )

    assert evaluation.decision == decision
    assert evaluation.incumbent_validation.hardware_probabilities == (
        evaluation.hardware_report.validation.hardware_probabilities
    )
    assert evaluation.candidate_improvement == pytest.approx(
        evaluation.incumbent_hardware_total_variation
        - evaluation.candidate_hardware_total_variation
    )
    assert evaluation.candidate_improvement_error_radius == pytest.approx(
        2.0 * evaluation.finite_shot_tv_radius
    )
    assert evaluation.to_dict()["decision"] == decision
    assert len(evaluation.trial_identity) == 64


def test_candidate_trial_rejects_changed_circuit_and_non_authoritative_program():
    trial, _ = _trial((0.7, 0.0, 0.0, 0.3), (0.98, 0.0, 0.0, 0.02))
    receipt, result = _result(trial, {"00": 1024})

    with pytest.raises(RuntimeError, match="circuit does not match"):
        trial.validate_result(
            result,
            receipt=receipt,
            circuit=fq.Circuit(2).x(0),
        )

    missing_program = replace(
        result,
        metadata=build_result_metadata(receipt, {"chip": "Shenglian"}),
    )
    with pytest.raises(RuntimeError, match="authoritative executed program"):
        trial.validate_result(
            missing_program,
            receipt=receipt,
            circuit=fq.Circuit(2).h(0).cx(0, 1),
        )


def test_candidate_trial_requires_same_target_and_later_snapshot():
    incumbent = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-15 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    earlier = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-14 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="must be later"):
        prepare_candidate_trial(
            incumbent,
            earlier,
            circuit,
            name="candidate-bell",
            shots=1024,
        )

    other = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-16 10:30:00"),
        target="quafu:Baihua",
        qubits=(20, 27),
    )
    with pytest.raises(ValueError, match="one target and mapping"):
        prepare_candidate_trial(
            incumbent,
            other,
            circuit,
            name="candidate-bell",
            shots=1024,
        )

    incumbent_acme = replace(
        incumbent,
        snapshot=replace(incumbent.snapshot, provider="acme"),
    )
    candidate_acme = replace(
        earlier,
        snapshot=replace(earlier.snapshot, provider="acme"),
    )
    with pytest.raises(ValueError, match="currently require Quafu"):
        prepare_candidate_trial(
            candidate_acme,
            incumbent_acme,
            circuit,
            name="candidate-bell",
            shots=1024,
        )


def test_complete_quafu_candidate_example_executes_offline(monkeypatch, tmp_path):
    incumbent = fq.twin.from_quafu_chip_info(
        _chip_info("2026-08-14 10:30:00"),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    incumbent_path = tmp_path / "twin-incumbent.json"
    fq.twin.dump_twin(incumbent, incumbent_path)

    class Provider:
        provider = "quafu"

        def __init__(self):
            self.qasm = ""
            self.submissions = 0

        def fetch_chip_info(self, chip):
            assert chip == "Shenglian"
            return _chip_info("2026-08-15 10:30:00")

        def submit_qasm(self, qasm, *, chip, name, shots, target_qubits):
            self.qasm = qasm
            self.submissions += 1
            digest = hashlib.sha256(qasm.encode()).hexdigest()
            return ProviderTaskHandle(
                provider="quafu",
                task_id="candidate-example-task",
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
                    {"chip": "Shenglian", "transpiled": self.qasm},
                ),
            )

    provider = Provider()
    monkeypatch.setenv("QUAFU_API_TOKEN", "offline-test-token")
    monkeypatch.setattr(quafu_twin_candidate, "QuafuProvider", lambda: provider)
    monkeypatch.setattr(quafu_twin_candidate, "INCUMBENT_PATH", incumbent_path)

    quafu_twin_candidate.main()

    assert provider.submissions == 1
