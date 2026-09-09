"""Resident Jiuding workspace executor behavior without platform mutation."""

import base64
import io

import pytest
import torch

import flagquantum as fq
from flagquantum.remote.compute import _workspace_executor as executor
from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


def test_cpu_executor_runs_bell_state_and_reports_actual_path():
    request = {
        "schema": executor.SCHEMA,
        "version": executor.VERSION,
        "operation": "statevector",
        "request_id": "bell",
        "target": "jiuding:cpu",
        "program": fq.Circuit(2).h(0).cx(0, 1).to_ir().to_dict(),
    }

    response = executor.execute(request, target="jiuding:cpu", device="cpu")

    encoded = response["state"]
    raw = base64.b64decode(encoded["data"])
    state = torch.frombuffer(bytearray(raw), dtype=torch.float32).view(torch.complex64)
    expected = torch.tensor([2**-0.5, 0, 0, 2**-0.5], dtype=torch.complex64)
    torch.testing.assert_close(state, expected)
    assert response["request_id"] == "bell"
    assert response["evidence"]["device"] == "cpu"
    assert response["evidence"]["cpu_fallback_used"] is False


def test_protocol_round_trip_and_size_limit():
    stream = io.BytesIO()
    executor.write_message(stream, {"ok": True})
    stream.seek(0)
    assert executor.read_message(stream) == {"ok": True}

    stream = io.BytesIO((executor.MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(ValueError, match="size"):
        executor.read_message(stream)


def test_executor_fails_closed_on_target_mismatch():
    request = {
        "schema": executor.SCHEMA,
        "version": executor.VERSION,
        "operation": "statevector",
        "target": "jiuding:gpu/NVIDIA_A100-SXM4-40GB",
        "program": fq.Circuit(1).to_ir().to_dict(),
    }
    with pytest.raises(ValueError, match="does not match"):
        executor.execute(request, target="jiuding:cpu", device="cpu")


def test_ssh_command_discards_unapproved_platform_options():
    client = JiudingClient(workspace="test")
    client._workspace = {
        "id": "workspace-id",
        "name": "test",
    }
    from unittest.mock import Mock

    client._workspace_record = Mock(
        return_value={
            "SSHLogin": (
                "ssh -o ProxyCommand=unsafe -CAXY "
                "worker.user@ssh.platform-multi.baai.ac.cn -p 2222"
            )
        }
    )

    command = client._ssh_base()
    assert client._ssh_base() == command

    assert "ProxyCommand=unsafe" not in command
    assert command[-3:] == ["-p", "2222", "worker.user@ssh.platform-multi.baai.ac.cn"]
    assert any(item.startswith("ControlPath=/tmp/fq-jiuding-") for item in command)
    client._workspace_record.assert_called_once_with("workspace-id")


def test_run_statevector_returns_normal_execution_result(monkeypatch):
    client = JiudingClient(workspace="test")
    client._workspace = {
        "id": "workspace-id",
        "name": "test",
        "acceleratorModel": "",
    }
    monkeypatch.setattr(client, "start_executor", lambda **_: {"ok": True})
    data = (
        torch.tensor([1, 0], dtype=torch.complex64)
        .view(torch.float32)
        .numpy()
        .tobytes()
    )
    monkeypatch.setattr(
        client,
        "_executor_request",
        lambda *_, **__: {
            "ok": True,
            "state": {
                "dtype": "complex64",
                "shape": [2],
                "data": base64.b64encode(data).decode(),
            },
            "evidence": {
                "device": "cpu",
                "elapsed_seconds": 0.01,
                "accelerator": "",
                "cpu_fallback_used": False,
            },
        },
    )

    result = client.run_statevector(fq.Circuit(1), target="jiuding:cpu")

    torch.testing.assert_close(
        result.to_statevector(), torch.tensor([1, 0], dtype=torch.complex64)
    )
    assert result.runtime["execution_path"] == "jiuding_workspace_executor"
    assert result.provenance["cpu_fallback_used"] is False


def test_default_workspace_client_is_reused(monkeypatch):
    from flagquantum.remote.compute import jiuding

    jiuding._DEFAULT_CLIENTS.clear()
    monkeypatch.setenv("JIUDING_WORKSPACE", "flagquantum-runtime")
    clients = []

    def execute(client, program, *, target):
        clients.append(client)
        return program

    monkeypatch.setattr(JiudingClient, "run_statevector", execute)
    circuit = fq.Circuit(1)

    assert jiuding.run_statevector(circuit, target="jiuding:gpu") is circuit
    assert jiuding.run_statevector(circuit, target="jiuding:gpu") is circuit
    assert clients[0] is clients[1]
    assert clients[0].workspace_name == "flagquantum-runtime"
    jiuding._DEFAULT_CLIENTS.clear()


def test_start_executor_reuses_verified_health(monkeypatch):
    client = JiudingClient(workspace="test")
    client._workspace = {
        "id": "workspace-id",
        "name": "test",
        "acceleratorModel": "",
    }
    health = {
        "ok": True,
        "target": "jiuding:cpu",
        "device": "cpu",
        "cuda_available": False,
    }
    client._executor_health[57621] = health
    monkeypatch.setattr(
        client,
        "_executor_request",
        lambda *_, **__: pytest.fail("cached executor must not be probed again"),
    )

    assert client.start_executor(target="jiuding:cpu") == health


def test_workspace_restart_discards_cached_transport(monkeypatch):
    from unittest.mock import Mock

    client = JiudingClient(workspace="test")
    client._workspace = {"id": "old"}
    client._ssh_command = ("ssh", "old")
    process = Mock()
    process.poll.return_value = None
    client._executor_channels[57621] = process
    client._executor_health[57621] = {"ok": True}

    client._reset_workspace_connection()

    assert client._workspace is None
    assert client._ssh_command is None
    assert client._executor_channels == {}
    assert client._executor_health == {}
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=1)
