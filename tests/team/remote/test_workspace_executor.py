"""Resident Jiuding workspace executor behavior without platform mutation."""

import base64
import io
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.remote.compute import _workspace_executor as executor
from flagquantum.remote.compute.jiuding import JiudingClient
from flagquantum.runtime.result import ExecutionResult

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


def test_cpu_executor_returns_measurements_without_statevector():
    from flagquantum.observables import lower_outputs

    ir = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    measurements = lower_outputs(
        (
            fq.probabilities(),
            fq.expectation(fq.X(0) @ fq.X(1), name="xx"),
        ),
        n_wires=2,
        shots=None,
    )
    assert measurements is not None
    request = {
        "schema": executor.SCHEMA,
        "version": executor.VERSION,
        "operation": "measurements",
        "request_id": "bell-measurements",
        "target": "jiuding:cpu",
        "program": replace(ir, measurements=measurements).to_dict(),
    }

    response = executor.execute(request, target="jiuding:cpu", device="cpu")

    assert "state" not in response
    probabilities, expectation = response["measurements"]
    probability_values = torch.frombuffer(
        bytearray(base64.b64decode(probabilities["value"]["data"])),
        dtype=torch.float32,
    ).reshape(probabilities["value"]["shape"])
    expectation_value = torch.frombuffer(
        bytearray(base64.b64decode(expectation["value"]["data"])),
        dtype=torch.float32,
    ).reshape(expectation["value"]["shape"])
    torch.testing.assert_close(probability_values, torch.tensor([[0.5, 0.0, 0.0, 0.5]]))
    torch.testing.assert_close(expectation_value, torch.tensor([1.0]))


def test_cpu_executor_returns_samples_and_counts_without_statevector():
    from flagquantum.observables import lower_outputs

    ir = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    measurements = lower_outputs(
        (fq.samples(wires=(0, 1)), fq.counts(wires=(0, 1))),
        n_wires=2,
        shots=32,
        seed=7,
    )
    assert measurements is not None
    response = executor.execute(
        {
            "schema": executor.SCHEMA,
            "version": executor.VERSION,
            "operation": "measurements",
            "request_id": "bell-sampling",
            "target": "jiuding:cpu",
            "program": replace(ir, measurements=measurements).to_dict(),
        },
        target="jiuding:cpu",
        device="cpu",
    )

    assert "state" not in response
    samples, counts = response["measurements"]
    assert samples["value"]["dtype"] == "int64"
    assert samples["value"]["shape"] == [1, 32, 2]
    count_values = {
        entry["outcome"]: entry["count"] for entry in counts["value"]["counts"][0]
    }
    assert sum(count_values.values()) == 32
    assert set(count_values) <= {"00", "11"}
    assert response["evidence"]["statevector_transferred"] is False
    assert response["evidence"]["counts_aggregation"] == "host"


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

    def execute(client, program, *, target, outputs=None, shots=None):
        clients.append(client)
        return program

    monkeypatch.setattr(JiudingClient, "run", execute)
    circuit = fq.Circuit(1)

    assert jiuding.run(circuit, target="jiuding:gpu") is circuit
    assert jiuding.run(circuit, target="jiuding:gpu") is circuit
    assert clients[0] is clients[1]
    assert clients[0].workspace_name == "flagquantum-runtime"
    jiuding._DEFAULT_CLIENTS.clear()


def test_client_reconstructs_remote_measurement_results(monkeypatch):
    client = JiudingClient(workspace="test")
    value = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    encoded = base64.b64encode(value.numpy().tobytes()).decode()
    monkeypatch.setattr(
        client,
        "_execute",
        lambda *_, **__: (
            {
                "measurements": [
                    {
                        "kind": "probabilities",
                        "wires": [0],
                        "value": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "data": encoded,
                        },
                        "shots": None,
                        "metadata": {"fq_output_index": 0},
                        "statistics": {},
                    }
                ],
                "evidence": {
                    "device": "cuda:0",
                    "elapsed_seconds": 0.01,
                    "accelerator": "A100",
                    "cpu_fallback_used": False,
                },
            },
            "jiuding:gpu/NVIDIA_A100-SXM4-40GB",
        ),
    )

    result = client.run(
        fq.Circuit(1),
        target="jiuding:gpu",
        outputs=fq.probabilities(),
    )

    assert isinstance(result, ExecutionResult)
    torch.testing.assert_close(result.probabilities, value)
    assert result.state is None
    assert result.runtime["device"] == "cuda:0"


def test_client_reconstructs_remote_samples_and_counts(monkeypatch):
    client = JiudingClient(workspace="test")
    samples = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int64)
    encoded = base64.b64encode(samples.numpy().tobytes()).decode()
    monkeypatch.setattr(
        client,
        "_execute",
        lambda *_, **__: (
            {
                "measurements": [
                    {
                        "kind": "sample",
                        "wires": [0, 1],
                        "value": {
                            "dtype": "int64",
                            "shape": [1, 2, 2],
                            "data": encoded,
                        },
                        "shots": 2,
                        "metadata": {"fq_output_index": 0},
                        "statistics": {},
                    },
                    {
                        "kind": "counts",
                        "wires": [0, 1],
                        "value": {
                            "counts": [
                                [
                                    {"outcome": 0, "count": 1},
                                    {"outcome": 3, "count": 1},
                                ]
                            ]
                        },
                        "shots": 2,
                        "metadata": {"fq_output_index": 1},
                        "statistics": {},
                    },
                ],
                "evidence": {
                    "device": "cuda:0",
                    "elapsed_seconds": 0.01,
                    "accelerator": "A100",
                    "cpu_fallback_used": False,
                },
            },
            "jiuding:gpu/NVIDIA_A100-SXM4-40GB",
        ),
    )

    result = client.run(
        fq.Circuit(2),
        target="jiuding:gpu",
        outputs=(fq.samples(), fq.counts()),
        shots=2,
    )

    torch.testing.assert_close(result.require_samples(), samples)
    assert result.counts == [{0: 1, 3: 1}]
    assert result.state is None


def test_client_requires_outputs_for_shots():
    client = JiudingClient(workspace="test")

    with pytest.raises(TypeError, match="requires fq.samples"):
        client.run(fq.Circuit(1), target="jiuding:cpu", shots=4)
    with pytest.raises(ValueError, match="requires shots"):
        client.run(
            fq.Circuit(1),
            target="jiuding:cpu",
            outputs=fq.samples(),
        )


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
