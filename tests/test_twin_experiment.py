"""Identity-chain tests for QPU digital-twin hardware validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    QuafuProvider,
    build_result_metadata,
)
from flagquantum.twin import QPUDigitalTwin, TwinExperiment

SUBMITTED_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[2];
h q[3];
cx q[3],q[4];
measure q[3] -> c[0];
measure q[4] -> c[1];
"""
FIXTURES = Path(__file__).parent / "fixtures"


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


def _experiment(
    *,
    submitted_qasm=SUBMITTED_QASM,
    shots=1024,
    backend_name="Baihua",
    circuit=None,
    physical_qubits=(3, 4),
):
    twin = QPUDigitalTwin.from_quafu_chip_info(
        _chip_info(), backend_name=backend_name, physical_qubits=physical_qubits
    )
    return TwinExperiment.prepare(
        twin,
        circuit if circuit is not None else fq.Circuit(2).h(0).cx(0, 1),
        submitted_qasm=submitted_qasm,
        name="frozen-bell",
        shots=shots,
    )


def _handle(experiment, *, digest=None):
    program_identity = digest or experiment.submitted_qasm_identity
    return ProviderTaskHandle(
        provider="quafu",
        task_id="task-42",
        backend_name=experiment.backend_name,
        payload={
            "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": program_identity,
            "deployment_artifact_sha256": program_identity,
            "submitted_qasm_sha256": program_identity,
        },
    )


def _result(experiment, *, executed_qasm=SUBMITTED_QASM):
    handle = _handle(experiment)
    metadata = build_result_metadata(
        handle,
        {} if executed_qasm is None else {"transpiled": executed_qasm},
    )
    return DeploymentResult(
        handle=handle,
        counts={"00": 512, "11": 512},
        shots=1024,
        metadata=metadata,
    )


def test_experiment_binds_prediction_program_receipt_and_result():
    experiment = _experiment()

    class Provider:
        provider = "quafu"

        def submit_qasm(self, qasm, *, chip, name, shots):
            assert (qasm, chip, name, shots) == (
                SUBMITTED_QASM,
                "Baihua",
                "frozen-bell",
                1024,
            )
            return _handle(experiment)

    handle = experiment.submit(Provider())
    report = experiment.validate_result(_result(experiment), receipt=handle)

    assert (
        handle.payload["submitted_qasm_sha256"]
        == hashlib.sha256(SUBMITTED_QASM.encode()).hexdigest()
    )
    assert report.task_id == "task-42"
    assert report.executed_program_matches_submission is True
    assert report.validation_scope == "retrospective_diagnostic"
    assert report.to_dict()["validation_scope"] == "retrospective_diagnostic"
    assert report.to_dict()["executed_program_matches_submission"] is True
    assert report.validation.shots == 1024
    assert len(report.identity) == 64


def test_rewritten_or_missing_executed_program_fails_closed():
    experiment = _experiment()

    receipt = _handle(experiment)
    rewritten = experiment.validate_result(
        _result(experiment, executed_qasm=SUBMITTED_QASM + "\n"), receipt=receipt
    )
    missing = experiment.validate_result(
        _result(experiment, executed_qasm=None), receipt=receipt
    )

    assert rewritten.executed_program_matches_submission is False
    assert missing.executed_program_matches_submission is False
    assert rewritten.validation_scope == "retrospective_diagnostic"
    assert missing.validation_scope == "retrospective_diagnostic"
    assert rewritten.validation.shots == missing.validation.shots == 1024


def test_real_quafu_response_shape_distinguishes_submitted_and_executed_qasm():
    response = json.loads(
        (FIXTURES / "quafu_result_service_transpile.json").read_text(encoding="utf-8")
    )
    # Sanitized from a live Shenglian compile=False probe on 2026-09-09.
    experiment = _experiment(
        submitted_qasm=response["circuit"],
        shots=response["shots"],
        backend_name=response["chip"],
        circuit=fq.Circuit(1).h(0),
        physical_qubits=(3,),
    )
    receipt = _handle(experiment)

    class Transport:
        def get_json(self, url, headers, timeout):
            return response

    provider = QuafuProvider(base_url="https://quafu.test", transport=Transport())
    result = provider.fetch_result(receipt)
    report = experiment.validate_result(result, receipt=receipt)

    assert result.metadata["circuit"] == experiment.submitted_qasm
    assert result.metadata["transpiled"] != experiment.submitted_qasm
    assert (
        report.executed_qasm_identity
        == hashlib.sha256(response["transpiled"].encode()).hexdigest()
    )
    assert report.executed_program_matches_submission is False
    assert report.validation_scope == "retrospective_diagnostic"
    assert report.validation.shots == response["shots"]


def test_experiment_rejects_receipt_or_result_from_another_program():
    experiment = _experiment()

    class Provider:
        provider = "quafu"

        def submit_qasm(self, qasm, *, chip, name, shots):
            return _handle(experiment, digest="0" * 64)

    with pytest.raises(RuntimeError, match="receipt"):
        experiment.submit(Provider())

    foreign = _result(experiment)
    foreign.handle.payload["submitted_qasm_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="submitted QASM"):
        experiment.validate_result(foreign, receipt=foreign.handle)


def test_experiment_rejects_result_from_another_task_receipt():
    experiment = _experiment()
    result = _result(experiment)
    other_receipt = ProviderTaskHandle(
        provider=result.handle.provider,
        task_id="task-99",
        backend_name=result.handle.backend_name,
        payload=result.handle.payload,
    )

    with pytest.raises(RuntimeError, match="task receipt"):
        experiment.validate_result(result, receipt=other_receipt)


def test_experiment_rejects_provider_result_reporting_another_backend():
    experiment = _experiment()
    receipt = _handle(experiment)
    result = DeploymentResult(
        handle=receipt,
        counts={"00": 512, "11": 512},
        shots=1024,
        metadata=build_result_metadata(
            receipt, {"chip": "Dongling", "transpiled": SUBMITTED_QASM}
        ),
    )

    with pytest.raises(RuntimeError, match="different QPU target"):
        experiment.validate_result(result, receipt=receipt)
