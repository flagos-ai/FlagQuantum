"""Prospective connected-region validation-suite scenarios."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import flagquantum as fq
from examples.remote import quafu_twin_region_validation_suite
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
            "Q20": {
                "T1": 41.0,
                "T2": 61.0,
                "fidelity": 0.997,
                "length": 6.4e-8,
            },
            "Q27": {
                "T1": 40.0,
                "T2": 60.0,
                "fidelity": 0.998,
                "length": 6.4e-8,
            },
            "Q34": {
                "T1": 42.0,
                "T2": 62.0,
                "fidelity": 0.996,
                "length": 6.4e-8,
            },
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
        supported_operations=("h", "cx", "rx"),
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
        maximum_circuit_depth=4,
    )
    return twin, support


def _region_twin() -> fq.twin.TwinRegionModel:
    return fq.twin.compose_region_twin((_cell((20, 27)), _cell((27, 34))))


def _circuits():
    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).h(1).cx(1, 2),
        fq.Circuit(3).h(0).cx(0, 1).h(2),
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


def _result(submission: fq.twin.TwinSubmission, *, offset: int):
    counts = {"000": 512 - offset, "111": 512 + offset}
    return DeploymentResult(
        handle=submission.receipt,
        counts=counts,
        shots=1024,
        metadata=build_result_metadata(
            submission.receipt,
            {"transpiled": submission.experiment.submitted_qasm},
        ),
    )


def _suite_inputs(suite: fq.twin.TwinRegionValidationSuite):
    submissions = []
    results = []
    task_index = 0
    for experiment in suite.experiments:
        for repetition in range(suite.repetitions):
            task_index += 1
            submission = _submission(experiment, f"region-suite-{task_index}")
            submissions.append(submission)
            results.append(_result(submission, offset=repetition * 8))
    return tuple(submissions), tuple(results)


def test_region_validation_suite_freezes_plan_without_submitting() -> None:
    region_twin = _region_twin()

    suite = fq.twin.prepare_region_validation_suite(
        region_twin,
        _circuits(),
        physical_qubits=(20, 27, 34),
        name="regional-workloads",
        shots=1024,
        repetitions=2,
    )

    assert suite.region_identity == region_twin.identity
    assert suite.circuit_count == 3
    assert suite.planned_task_count == 6
    assert suite.planned_shots == 6144
    assert suite.directed_couplers == ((20, 27), (27, 34))
    assert tuple(item.name for item in suite.experiments) == (
        "regional-workloads-01",
        "regional-workloads-02",
        "regional-workloads-03",
    )


def test_region_validation_suite_builds_simultaneous_exact_support() -> None:
    region_twin = _region_twin()
    circuits = _circuits()
    suite = fq.twin.prepare_region_validation_suite(
        region_twin,
        circuits,
        physical_qubits=(20, 27, 34),
        name="regional-workloads",
        shots=1024,
        repetitions=2,
    )
    submissions, results = _suite_inputs(suite)

    evaluation = suite.validate_results(
        submissions,
        results,
        circuits=circuits,
        confidence_level=0.95,
    )
    support = evaluation.to_circuit_support()

    assert evaluation.circuit_count == 3
    assert evaluation.task_count == 6
    assert evaluation.total_shots == 6144
    assert evaluation.confidence_level == pytest.approx(0.95)
    assert all(
        item.confidence_level == pytest.approx(1.0 - 0.05 / 3.0)
        for item in evaluation.validation_series
    )
    assert evaluation.simultaneous_tv_error_bound == max(
        item.verified_tv_error_bound for item in evaluation.validation_series
    )
    assert 0.0 <= evaluation.mean_twin_qpu_agreement <= 1.0
    assert 0.0 <= evaluation.mean_ideal_qpu_agreement <= 1.0
    assert evaluation.mean_qpu_repeatability == pytest.approx(0.9921875)
    assert support.directed_couplers == ((20, 27), (27, 34))
    assert support.evidence.confidence_level == pytest.approx(0.95)
    assert support.evidence.verified_circuit_identities == tuple(
        circuit.to_ir().content_hash for circuit in circuits
    )
    assert all(
        support.evidence_report(region_twin.twin, circuit).status
        == "exact_circuit_verified"
        for circuit in circuits
    )


def test_region_validation_suite_fails_closed_on_partial_or_reused_tasks() -> None:
    circuits = _circuits()
    suite = fq.twin.prepare_region_validation_suite(
        _region_twin(),
        circuits,
        physical_qubits=(20, 27, 34),
        name="regional-workloads",
        shots=1024,
        repetitions=2,
    )
    submissions, results = _suite_inputs(suite)

    with pytest.raises(ValueError, match="every planned suite task"):
        suite.validate_results(submissions[:-1], results[:-1], circuits=circuits)

    duplicate = replace(
        submissions[-1],
        receipt=replace(
            submissions[-1].receipt,
            task_id=submissions[0].receipt.task_id,
        ),
    )
    with pytest.raises(ValueError, match="distinct QPU tasks"):
        suite.validate_results(
            (*submissions[:-1], duplicate),
            (*results[:-1], _result(duplicate, offset=0)),
            circuits=circuits,
        )


def test_region_validation_suite_round_trip_is_private_and_create_once(
    tmp_path,
) -> None:
    suite = fq.twin.prepare_region_validation_suite(
        _region_twin(),
        _circuits(),
        physical_qubits=(20, 27, 34),
        name="regional-workloads",
        shots=1024,
        repetitions=2,
    )
    destination = tmp_path / "region-suite.json"

    fq.twin.dump_region_validation_suite(suite, destination)
    original = destination.read_bytes()
    restored = fq.twin.load_region_validation_suite(destination)
    fq.twin.dump_region_validation_suite(suite, destination)

    assert restored == suite
    assert restored.identity == suite.identity
    assert destination.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o600


def test_region_validation_suite_rejects_small_or_duplicate_workloads() -> None:
    region_twin = _region_twin()
    circuit = _circuits()[0]

    with pytest.raises(ValueError, match="at least two circuits"):
        fq.twin.prepare_region_validation_suite(
            region_twin,
            (circuit,),
            physical_qubits=(20, 27, 34),
            name="too-small",
            shots=1024,
        )
    with pytest.raises(ValueError, match="distinct circuits"):
        fq.twin.prepare_region_validation_suite(
            region_twin,
            (circuit, circuit),
            physical_qubits=(20, 27, 34),
            name="duplicates",
            shots=1024,
        )


def test_complete_region_suite_example_is_checkpointed_and_executable(
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
        "SUITE_PATH": tmp_path / "region-suite.json",
        "SUPPORT_PATH": tmp_path / "region-support.json",
    }
    fq.twin.dump_twin(twin_a, paths["CELL_A_TWIN"])
    fq.twin.dump_circuit_support(support_a, paths["CELL_A_SUPPORT"])
    fq.twin.dump_twin(twin_b, paths["CELL_B_TWIN"])
    fq.twin.dump_circuit_support(support_b, paths["CELL_B_SUPPORT"])
    for name, path in paths.items():
        monkeypatch.setattr(quafu_twin_region_validation_suite, name, path)
    monkeypatch.setattr(
        quafu_twin_region_validation_suite,
        "SUBMISSION_PREFIX",
        str(tmp_path / "submission"),
    )
    monkeypatch.setattr(
        quafu_twin_region_validation_suite,
        "SERIES_PREFIX",
        str(tmp_path / "series"),
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
            task_id = f"example-suite-{self.submissions}"
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
    quafu_twin_region_validation_suite.prepare()
    suite = fq.twin.load_region_validation_suite(paths["SUITE_PATH"])
    assert suite.planned_task_count == 6
    assert provider.submissions == 0

    for index in range(1, suite.planned_task_count + 1):
        quafu_twin_region_validation_suite.submit_one(provider, index)
    with pytest.raises(FileExistsError, match="refusing to resubmit"):
        quafu_twin_region_validation_suite.submit_one(provider, 1)
    assert provider.submissions == 6

    quafu_twin_region_validation_suite.evaluate(provider)

    restored = fq.twin.load_circuit_support(paths["SUPPORT_PATH"])
    assert len(restored.evidence.verified_circuit_identities) == 3
    assert tuple(tmp_path.glob("series-*.json"))
