from __future__ import annotations

import json
from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)
from flagquantum.twin import TwinPrediction

pytestmark = pytest.mark.integration


def _chip_info(captured_at: str, fidelity: float):
    return {
        "calibration_time": captured_at,
        "basis_gates": ["h", "x", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {"T1": 41.0, "T2": 61.0, "fidelity": fidelity, "length": 6.4e-8},
            "Q27": {"T1": 40.0, "T2": 60.0, "fidelity": fidelity, "length": 6.4e-8},
            "Q34": {"T1": 42.0, "T2": 62.0, "fidelity": fidelity, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": fidelity,
                "length": 2.24e-7,
            },
            "C1": {
                "qubits_index": [27, 34],
                "fidelity": fidelity,
                "length": 2.24e-7,
            },
        },
    }


def _cell(captured_at: str, fidelity: float, qubits: tuple[int, int]):
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(captured_at, fidelity),
        target="quafu:Shenglian",
        qubits=qubits,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=qubits,
        supported_operations=("h", "x", "cx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(circuit.to_ir().content_hash,),
        evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    support = fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=5,
    )
    return twin, support


def _region(captured_at: str, fidelity: float) -> fq.twin.TwinRegionModel:
    return fq.twin.compose_region_twin(
        (
            _cell(captured_at, fidelity, (20, 27)),
            _cell(captured_at, fidelity, (27, 34)),
        )
    )


def _reference_circuits():
    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).x(0).cx(0, 1),
    )


def _holdout_circuits():
    return (
        fq.Circuit(3).h(1).cx(1, 2),
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2).x(2),
    )


def _study() -> fq.twin.TwinRegionCandidateHoldoutStudy:
    return fq.twin.prepare_region_candidate_holdout(
        _region("2026-08-14 10:30:00", 0.96),
        _region("2026-08-15 10:30:00", 0.99),
        _reference_circuits(),
        _holdout_circuits(),
        physical_qubits=(20, 27, 34),
        name="regional-candidate",
        shots=4096,
    )


def _prediction(base: TwinPrediction, probabilities: tuple[float, ...]):
    distance = 0.5 * sum(
        abs(ideal - predicted)
        for ideal, predicted in zip(
            base.ideal_probabilities, probabilities, strict=True
        )
    )
    return replace(
        base,
        twin_probabilities=probabilities,
        total_variation_from_ideal=distance,
    )


def _with_predictions(
    study: fq.twin.TwinRegionCandidateHoldoutStudy,
    *,
    reference_incumbent: tuple[float, ...] = (0.70, 0, 0, 0, 0, 0, 0, 0.30),
    reference_candidate: tuple[float, ...] = (0.98, 0, 0, 0, 0, 0, 0, 0.02),
):
    def adjust(suite, incumbent_probabilities, candidate_probabilities):
        return replace(
            suite,
            trials=tuple(
                replace(
                    trial,
                    incumbent_prediction=_prediction(
                        trial.incumbent_prediction, incumbent_probabilities
                    ),
                    experiment=replace(
                        trial.experiment,
                        prediction=_prediction(
                            trial.experiment.prediction, candidate_probabilities
                        ),
                    ),
                )
                for trial in suite.trials
            ),
        )

    return replace(
        study,
        reference_suite=adjust(
            study.reference_suite,
            reference_incumbent,
            reference_candidate,
        ),
        holdout_suite=adjust(
            study.holdout_suite,
            (0.70, 0, 0, 0, 0, 0, 0, 0.30),
            (0.98, 0, 0, 0, 0, 0, 0, 0.02),
        ),
    )


def _inputs(suite: fq.twin.TwinCandidateSuite, prefix: str):
    submissions = []
    results = []
    for index, trial in enumerate(suite.trials, start=1):
        experiment = trial.experiment
        receipt = ProviderTaskHandle(
            provider="quafu",
            task_id=f"{prefix}-{index}",
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
        submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
        bound_receipt = submission.receipt
        result = DeploymentResult(
            handle=bound_receipt,
            counts={"000": 4096},
            shots=4096,
            metadata=build_result_metadata(
                bound_receipt,
                {"transpiled": experiment.submitted_qasm},
            ),
        )
        submissions.append(submission)
        results.append(result)
    return tuple(submissions), tuple(results)


def _evaluate(study: fq.twin.TwinRegionCandidateHoldoutStudy):
    reference_submissions, reference_results = _inputs(
        study.reference_suite, "reference"
    )
    holdout_submissions, holdout_results = _inputs(study.holdout_suite, "holdout")
    return study.validate_results(
        reference_submissions,
        reference_results,
        holdout_submissions,
        holdout_results,
        reference_circuits=_reference_circuits(),
        holdout_circuits=_holdout_circuits(),
        confidence_level=0.95,
    )


def test_region_candidate_holdout_freezes_one_task_per_disjoint_circuit() -> None:
    study = _study()

    assert study.reference_circuit_count == 2
    assert study.holdout_circuit_count == 2
    assert study.planned_task_count == 4
    assert study.planned_shots == 16384
    assert set(
        trial.incumbent_prediction.circuit_identity
        for trial in study.reference_suite.trials
    ).isdisjoint(
        trial.incumbent_prediction.circuit_identity
        for trial in study.holdout_suite.trials
    )


def test_region_candidate_holdout_uses_holdout_as_upgrade_gate() -> None:
    evaluation = _evaluate(_with_predictions(_study()))

    assert evaluation.decision == "improved"
    assert evaluation.reference_evaluation.decision == "improved"
    assert evaluation.holdout_evaluation.decision == "improved"
    assert evaluation.circuit_count == 4
    assert evaluation.task_count == 4
    assert evaluation.total_shots == 16384
    assert evaluation.reference_evaluation.confidence_level == pytest.approx(0.975)
    assert evaluation.holdout_evaluation.confidence_level == pytest.approx(0.975)
    assert all(
        item.confidence_level == pytest.approx(0.9875)
        for item in evaluation.holdout_evaluation.evaluations
    )


def test_region_candidate_reference_degradation_vetoes_holdout_improvement() -> None:
    evaluation = _evaluate(
        _with_predictions(
            _study(),
            reference_incumbent=(0.98, 0, 0, 0, 0, 0, 0, 0.02),
            reference_candidate=(0.70, 0, 0, 0, 0, 0, 0, 0.30),
        )
    )

    assert evaluation.reference_evaluation.decision == "degraded"
    assert evaluation.holdout_evaluation.decision == "improved"
    assert evaluation.decision == "degraded"


def test_region_candidate_holdout_fails_closed_on_scope_and_leakage() -> None:
    incumbent = _region("2026-08-14 10:30:00", 0.96)
    candidate = _region("2026-08-15 10:30:00", 0.99)
    changed_scope = replace(
        candidate,
        region=replace(candidate.region, maximum_circuit_depth=4),
    )
    with pytest.raises(ValueError, match="share target, mapping, topology"):
        fq.twin.prepare_region_candidate_holdout(
            incumbent,
            changed_scope,
            _reference_circuits(),
            _holdout_circuits(),
            physical_qubits=(20, 27, 34),
            name="scope-change",
            shots=1024,
        )
    reference = _reference_circuits()
    with pytest.raises(ValueError, match="must be disjoint"):
        fq.twin.prepare_region_candidate_holdout(
            incumbent,
            candidate,
            reference,
            (reference[0], _holdout_circuits()[0]),
            physical_qubits=(20, 27, 34),
            name="leaked-circuit",
            shots=1024,
        )


def test_region_candidate_holdout_study_round_trip_is_create_once(tmp_path) -> None:
    study = _study()
    destination = tmp_path / "regional-candidate-holdout.json"

    fq.twin.dump_region_candidate_holdout_study(study, destination)
    fq.twin.dump_region_candidate_holdout_study(study, destination)
    restored = fq.twin.load_region_candidate_holdout_study(destination)

    assert restored == study
    assert restored.identity == study.identity
    assert destination.stat().st_mode & 0o777 == 0o600

    changed = replace(
        study,
        reference_suite=replace(
            study.reference_suite,
            trials=tuple(reversed(study.reference_suite.trials)),
        ),
    )
    with pytest.raises(ValueError):
        fq.twin.dump_region_candidate_holdout_study(changed, destination)


def test_region_candidate_holdout_evaluation_round_trip_is_create_once(
    tmp_path,
) -> None:
    evaluation = _evaluate(_with_predictions(_study()))
    destination = tmp_path / "regional-candidate-evaluation.json"

    fq.twin.dump_region_candidate_holdout_evaluation(evaluation, destination)
    fq.twin.dump_region_candidate_holdout_evaluation(evaluation, destination)
    restored = fq.twin.load_region_candidate_holdout_evaluation(destination)

    assert restored == evaluation
    assert restored.identity == evaluation.identity
    assert destination.stat().st_mode & 0o777 == 0o600

    different_destination = tmp_path / "occupied-evaluation.json"
    different_destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fq.twin.dump_region_candidate_holdout_evaluation(
            evaluation, different_destination
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update({"total_shots": 1}),
        lambda payload: payload.update({"total_shots": 16384.0}),
        lambda payload: payload["holdout_evaluation"].update(
            {"mean_candidate_improvement": 0.0}
        ),
        lambda payload: payload["reference_evaluation"]["evaluations"][0][
            "hardware_report"
        ]["counts"].update({"000": 4095}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.pop("task_count"),
    ),
)
def test_region_candidate_holdout_evaluation_rejects_noncanonical_payloads(
    tmp_path,
    mutation,
) -> None:
    evaluation = _evaluate(_with_predictions(_study()))
    payload = evaluation.to_dict()
    mutation(payload)
    destination = tmp_path / "tampered-evaluation.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        fq.twin.load_region_candidate_holdout_evaluation(destination)
