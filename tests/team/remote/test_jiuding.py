"""CPU-only scenarios for the experimental Jiuding task adapter."""

import json
from unittest.mock import Mock

import pytest

from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


def test_token_is_cached_and_refreshed_from_server_expiry(monkeypatch):
    monkeypatch.setenv("JIUDING_AK", "test-ak")
    monkeypatch.setenv("JIUDING_SK", "test-sk")
    client = JiudingClient()
    client._request = Mock(
        side_effect=[
            {"token": "first", "expiresIn": "604800"},
            {"token": "second", "expiresIn": "7200"},
        ]
    )
    assert client._auth() == {"AIRS-Token": "first"}
    assert client._auth() == {"AIRS-Token": "first"}
    assert client._request.call_count == 1
    client._expires = 0
    assert client._auth() == {"AIRS-Token": "second"}
    assert client._request.call_count == 2


def test_worker_returns_run_bound_json_and_rejects_nonfinite(tmp_path, monkeypatch):
    from flagquantum.remote.compute import _worker

    script, output = tmp_path / "entry.py", tmp_path / "result.json"
    script.write_text("def main(): return {'answer': 42}")
    monkeypatch.setattr("sys.argv", ["worker", str(script), str(output), "run"])
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    _worker.main()
    assert json.loads(output.read_text()) == {"run_id": "run", "value": {"answer": 42}}
    other = tmp_path / "invalid.json"
    script.write_text("def main(): return float('nan')")
    monkeypatch.setattr("sys.argv", ["worker", str(script), str(other), "other"])
    with pytest.raises(ValueError):
        _worker.main()
    assert not other.exists()


@pytest.fixture
def client():
    client = JiudingClient(workspace="test")
    client._workspace = dict(
        name="test",
        projId="p",
        projsetId="ps",
        queueId="q",
        clusterId="c",
        zoneId="z",
        queueStatus="QUEUE_STATUS_ACTIVE",
        resourceRegion="CUSTOMIZED_ACCELERATOR",
        creatorId="u",
        creatorName="user",
        storageInfo=[],
    )
    client._auth = Mock(return_value={"AIRS-Token": "test-token"})
    client._jobs = Mock(
        return_value=[dict(queueId="q", basicImage="image", imageRegion="PUBLIC")]
    )
    return client


def test_ambiguous_workspace_fails():
    client = JiudingClient(workspace="same")
    client._auth = Mock(return_value={})
    client._pages = Mock(return_value=[{"name": "same"}, {"name": "same"}])
    with pytest.raises(RuntimeError, match="unique"):
        client.workspace()


def test_workspace_discards_embedded_service_token():
    client = JiudingClient(workspace="test")
    client._auth = Mock(return_value={})
    fields = (
        "id",
        "name",
        "projId",
        "projsetId",
        "queueId",
        "queueName",
        "clusterId",
        "zoneId",
        "queueStatus",
        "resourceRegion",
        "creatorId",
        "creatorName",
    )
    workspace = dict.fromkeys(fields, "test")
    workspace["jupyterServiceToken"] = "secret"
    client._pages = Mock(return_value=[workspace])
    assert "secret" not in json.dumps(client.workspace())


def test_uncertain_create_is_not_retried(client, tmp_path):
    script = tmp_path / "task.py"
    script.write_text("def main(): return 42")
    receipt = tmp_path / "receipt.json"
    client._request = Mock(side_effect=RuntimeError("timeout"))
    with pytest.raises(RuntimeError, match="timeout"):
        client.submit(script, image="image", receipt=receipt)
    saved = json.loads(receipt.read_text())
    assert saved["submission"] == "unknown"
    assert "test-token" not in receipt.read_text()
    with pytest.raises(FileExistsError):
        client.submit(script, image="image", receipt=receipt)
    assert client._request.call_count == 1


def test_platform_success_requires_matching_scientific_result(client, tmp_path):
    path = tmp_path / "result.json"
    receipt = dict(run_id="run", result_path=str(path))
    client.status = Mock(return_value=[{"status": "Succeed"}])
    with pytest.raises(FileNotFoundError):
        client.result(receipt)
    path.write_text(json.dumps({"run_id": "another-run", "value": 45}))
    with pytest.raises(RuntimeError, match="does not belong"):
        client.result(receipt)
    path.write_text(json.dumps({"run_id": "run", "value": {"energy": -1}}))
    assert client.result(receipt) == {"energy": -1}


def test_empty_jobs_timeout_does_not_cancel_or_submit(client):
    client.status = Mock(return_value=[])
    client.cancel = Mock()
    client.submit = Mock()
    with pytest.raises(TimeoutError):
        client.result({}, timeout=0)
    client.cancel.assert_not_called()
    client.submit.assert_not_called()


def test_failed_job_is_not_reported_as_result(client):
    client.status = Mock(return_value=[{"status": "Failed"}])
    with pytest.raises(RuntimeError, match="did not succeed"):
        client.result({})


def test_different_queue_receipt_is_rejected(client):
    with pytest.raises(ValueError, match="different"):
        client.status(
            {"endpoint": client.endpoint, "queueId": "other", "experimentId": "e"}
        )


def test_http_and_redirect_targets_cannot_be_configured():
    for endpoint in (
        "http://example.com",
        "https://user:password@example.com",
        "https://example.com/path",
    ):
        with pytest.raises(ValueError):
            JiudingClient(endpoint=endpoint)


def test_submit_creates_then_launches_once_with_zero_gpu(client, tmp_path):
    script = tmp_path / "task.py"
    script.write_text("def main(): return 42")
    config = {"confId": "conf", "configName": "config1"}
    client._request = Mock(
        side_effect=[
            {"experimentId": "exp"},
            {
                "experimentSummaryInfos": [
                    {
                        "experimentId": "exp",
                        "advanceConfigInfos": [config],
                        "creatorId": "u",
                        "nativeCluster": "default-cluster",
                    }
                ]
            },
            {"jobId": "job"},
        ]
    )
    receipt = client.submit(script, image="image", receipt=tmp_path / "receipt.json")
    assert receipt["jobId"] == "job"
    calls = client._request.call_args_list
    assert [c.args[0] for c in calls] == [
        "/api/v1/experiment",
        "/api/v1/experiment/select",
        "/api/v1/job",
    ]
    resource = calls[0].args[1]["advanceConfigInfos"][0]["resourceConfigList"][0]
    assert resource["roleInfoList"][0]["resourceRequestDetail"]["acceleratorCount"] == 0
    assert "test-token" not in json.dumps(receipt)


def test_uncertain_launch_retains_experiment_and_never_recreates(client, tmp_path):
    script = tmp_path / "task.py"
    script.write_text("def main(): return 42")
    path = tmp_path / "receipt.json"
    client._request = Mock(
        side_effect=[
            {"experimentId": "exp"},
            {
                "experimentSummaryInfos": [
                    {
                        "experimentId": "exp",
                        "advanceConfigInfos": [
                            {"confId": "c", "configName": "config1"}
                        ],
                        "creatorId": "u",
                        "nativeCluster": "default-cluster",
                    }
                ]
            },
            RuntimeError("timeout"),
        ]
    )
    with pytest.raises(RuntimeError, match="timeout"):
        client.submit(script, image="image", receipt=path)
    saved = json.loads(path.read_text())
    assert saved["experimentId"] == "exp"
    assert saved["submission"] == "launch_unknown"
    with pytest.raises(FileExistsError):
        client.submit(script, image="image", receipt=path)
    assert client._request.call_count == 3


def test_cancel_only_active_jobs_without_archiving(client):
    client.status = Mock(
        return_value=[
            {"id": "active", "status": "Running"},
            {"id": "done", "status": "Succeed"},
        ]
    )
    client._request = Mock(return_value={})
    client.cancel({})
    assert client._request.call_count == 1
    assert client._request.call_args.args[:2] == (
        "/api/v1/job/cancel",
        {"jobIds": ["active"]},
    )
