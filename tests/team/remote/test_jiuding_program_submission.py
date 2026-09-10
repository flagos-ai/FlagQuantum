from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import torch

import flagquantum as fq
from flagquantum.remote.compute._program_job import (
    PROGRAM_JOB_SCHEMA,
    decode_program_result,
    execute_request,
)
from flagquantum.remote.compute._workspace_executor import (
    SCHEMA,
    VERSION,
    _encode_tensor,
)
from flagquantum.remote.compute.jiuding import JiudingClient


def _client() -> JiudingClient:
    client = JiudingClient(workspace="test")
    client._workspace = {
        "name": "test",
        "projId": "project",
        "projsetId": "project-set",
        "queueId": "queue",
        "queueName": "test-queue",
        "clusterId": "cluster",
        "zoneId": "zone",
        "queueStatus": "QUEUE_STATUS_ACTIVE",
        "resourceRegion": "CUSTOMIZED_ACCELERATOR",
        "creatorId": "user",
        "creatorName": "user",
        "acceleratorModel": "NVIDIA_A100-SXM4-40GB",
        "clusterName": "test-cluster",
        "storageInfo": [],
    }
    return client


def test_submit_program_builds_shared_bundle_without_user_script(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client()
    receipt_path = tmp_path / "bell.json"
    captured = {}

    def submit(script, **options):
        captured.update(script=script, **options)
        return {
            "run_id": "run-1",
            "result_path": str(tmp_path / "result.json"),
            "receipt": str(receipt_path),
        }

    monkeypatch.setattr(client, "submit", submit)
    output = fq.counts(wires=(0, 1))

    receipt = client.submit_program(
        fq.Circuit(2).h(0).cx(0, 1),
        target="jiuding:gpu",
        image="flagquantum-dev:v1",
        receipt=receipt_path,
        outputs=output,
        shots=1024,
        pythonpath=tmp_path,
    )

    request = json.loads(Path(receipt["program_request"]).read_text())
    assert request["schema"] == PROGRAM_JOB_SCHEMA
    assert request["operation"] == "measurements"
    assert request["target"] == "jiuding:gpu/NVIDIA_A100-SXM4-40GB"
    assert request["program"]["measurements"][0]["shots"] == 1024
    assert Path(captured["script"]).name == "bell.runner.py"
    assert captured["pythonpath"] == tmp_path
    assert receipt["submission_kind"] == "flagquantum_program"


def test_program_request_executes_through_standard_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": PROGRAM_JOB_SCHEMA,
                "version": "1.0",
                "target": "jiuding:cpu",
                "operation": "statevector",
                "program": fq.Circuit(1).h(0).to_ir().to_dict(),
            }
        )
    )
    expected = {
        "schema": SCHEMA,
        "version": VERSION,
        "ok": True,
        "request_id": "batch-program",
        "state": _encode_tensor(torch.tensor([1.0, 0.0], dtype=torch.complex64)),
        "evidence": {"device": "cpu", "target": "jiuding:cpu"},
    }
    execute = Mock(return_value=expected)
    monkeypatch.setattr(
        "flagquantum.remote.compute._workspace_executor.execute", execute
    )

    assert execute_request(request) == expected
    assert execute.call_args.kwargs == {"target": "jiuding:cpu", "device": "cpu"}


def test_program_result_is_a_normal_execution_result() -> None:
    response = {
        "schema": SCHEMA,
        "version": VERSION,
        "ok": True,
        "request_id": "batch-program",
        "state": _encode_tensor(
            torch.tensor([2**-0.5, 0, 0, 2**-0.5], dtype=torch.complex64)
        ),
        "evidence": {
            "device": "cuda:0",
            "elapsed_seconds": 0.01,
            "accelerator": "A100",
            "cpu_fallback_used": False,
            "target": "jiuding:gpu/NVIDIA_A100-SXM4-40GB",
        },
    }

    result = decode_program_result(
        response,
        requested_target="jiuding:gpu",
        selected_target="jiuding:gpu/NVIDIA_A100-SXM4-40GB",
    )

    assert result.to_statevector().shape == (4,)
    assert result.runtime["execution_path"] == "jiuding_batch_program"
    assert result.provenance["cpu_fallback_used"] is False


def test_client_result_decodes_program_receipt(tmp_path: Path, monkeypatch) -> None:
    client = _client()
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "value": {
                    "schema": SCHEMA,
                    "version": VERSION,
                    "ok": True,
                    "request_id": "batch-program",
                    "state": _encode_tensor(
                        torch.tensor([1.0, 0.0], dtype=torch.complex64)
                    ),
                    "evidence": {
                        "device": "cpu",
                        "target": "jiuding:cpu",
                        "cpu_fallback_used": False,
                    },
                },
                "worker": {"requested_gpus": 0},
            }
        )
    )
    receipt = {
        "run_id": "run-1",
        "result_path": str(result_path),
        "resources": {"target": "jiuding:cpu"},
    }
    monkeypatch.setattr(client, "status", Mock(return_value=[{"status": "Succeed"}]))

    result = client.result(receipt)

    assert result.to_statevector().shape == (2,)
    assert result.provenance["selected_target"] == "jiuding:cpu"


def test_submit_program_rejects_shots_without_sampling_output(tmp_path: Path) -> None:
    client = _client()

    try:
        client.submit_program(
            fq.Circuit(1).h(0),
            target="jiuding:cpu",
            image="flagquantum-dev:v1",
            receipt=tmp_path / "bad.json",
            shots=10,
        )
    except TypeError as error:
        assert "shots requires" in str(error)
    else:
        raise AssertionError("shots without an output must fail")
    assert list(tmp_path.iterdir()) == []


def test_submit_program_removes_bundle_after_preflight_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client()
    receipt_path = tmp_path / "failed.json"
    monkeypatch.setattr(
        client,
        "submit",
        Mock(side_effect=ValueError("image is unavailable")),
    )

    try:
        client.submit_program(
            fq.Circuit(1).h(0),
            target="jiuding:cpu",
            image="missing:v1",
            receipt=receipt_path,
        )
    except ValueError as error:
        assert "image is unavailable" in str(error)
    else:
        raise AssertionError("submission preflight failure must be returned")
    assert list(tmp_path.iterdir()) == []
