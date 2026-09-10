"""User-level Jiuding execution paths without external platform access."""

import json
import subprocess

import pytest
import torch

import flagquantum as fq
from flagquantum.remote.compute import _managed_program as managed
from flagquantum.remote.compute import _workspace_executor as executor
from flagquantum.remote.compute import jiuding
from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


def test_fq_run_executes_bell_measurements_through_jiuding(monkeypatch) -> None:
    monkeypatch.setenv("JIUDING_WORKSPACE", "golden-path")
    monkeypatch.setattr(jiuding, "_DEFAULT_CLIENTS", {})
    monkeypatch.setattr(JiudingClient, "start_executor", lambda self, **kwargs: None)

    def execute_locally(self, request, *, port, timeout):
        assert port == 57621
        assert timeout == 30
        return executor.execute(request, target=request["target"], device="cpu")

    monkeypatch.setattr(JiudingClient, "_executor_request", execute_locally)

    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        target="jiuding:cpu",
        outputs=(
            fq.probabilities(),
            fq.expectation(fq.X(0) @ fq.X(1), name="xx"),
        ),
    )

    torch.testing.assert_close(
        result.probabilities,
        torch.tensor([[0.5, 0.0, 0.0, 0.5]]),
    )
    torch.testing.assert_close(result.expectation("xx"), torch.tensor([1.0]))
    assert result.runtime["execution_path"] == "jiuding_workspace_executor"
    assert result.provenance["requested_target"] == "jiuding:cpu"
    assert result.provenance["selected_target"] == "jiuding:cpu"
    assert result.provenance["cpu_fallback_used"] is False


def test_submit_restore_and_read_managed_bell_job(monkeypatch) -> None:
    job_id = "1b64c2b7-3a7c-4feb-8687-9b18a892a0b8"
    client = JiudingClient(workspace="golden-path")
    client._workspace = {"name": "golden-path"}
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    submitted_commands = 0
    receipt = {}
    job_value = executor.execute(
        {
            "schema": executor.SCHEMA,
            "version": executor.VERSION,
            "operation": "statevector",
            "request_id": "batch-program",
            "target": "jiuding:cpu",
            "program": circuit.to_ir().to_dict(),
        },
        target="jiuding:cpu",
        device="cpu",
    )

    monkeypatch.setattr(managed, "_ensure_source", lambda client: "/managed/source")

    def run_ssh(client, arguments, **options):
        nonlocal submitted_commands, receipt
        stdout = b""
        if arguments[0] == "env":
            submitted_commands += 1
            payload = json.loads(arguments[-1])
            managed_fields = payload["_managed"]
            directory = payload["receipt"].rsplit("/", 1)[0]
            receipt = {
                **managed_fields,
                "jobId": job_id,
                "run_id": "run-1",
                "receipt": payload["receipt"],
                "result_path": f"{directory}/result.json",
                "resources": {"target": payload["target"]},
            }
            stdout = json.dumps(receipt).encode()
        elif job_id in arguments:
            stdout = json.dumps(receipt).encode()
        elif receipt.get("result_path") in arguments:
            payload = {"run_id": "run-1", "value": job_value}
            stdout = json.dumps(payload).encode()
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(managed, "_run_ssh", run_ssh)

    submitted = client.submit_program(
        circuit,
        target="jiuding:cpu",
        image="flagquantum-runtime:v1",
    )
    restored = client.restore_receipt(submitted["jobId"])
    monkeypatch.setattr(
        client,
        "status",
        lambda receipt: [{"status": "Succeed"}],
    )
    result = client.result(restored)

    assert submitted_commands == 1
    assert restored["jobId"] == job_id
    torch.testing.assert_close(
        result.to_statevector(),
        torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex64),
    )
    assert result.runtime["execution_path"] == "jiuding_batch_program"
