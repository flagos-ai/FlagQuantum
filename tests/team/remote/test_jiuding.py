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
    assert json.loads(output.read_text()) == {
        "run_id": "run",
        "value": {"answer": 42},
        "worker": {"requested_gpus": 0},
    }
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
        queueName="test-queue",
        clusterId="c",
        zoneId="z",
        queueStatus="QUEUE_STATUS_ACTIVE",
        resourceRegion="CUSTOMIZED_ACCELERATOR",
        creatorId="u",
        creatorName="user",
        acceleratorModel="NVIDIA_A100-SXM4-40GB",
        clusterName="airs-beijing-01",
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


def test_create_workspace_uses_explicit_non_privileged_configuration(client):
    client._catalog_image = Mock(
        return_value={
            "id": "private-image",
            "name": "flagquantum-runtime",
            "tag": "v0.2.0",
        }
    )
    client._request = Mock(return_value={"id": "workspace-id"})

    created = client.create_workspace(
        "flagquantum-dev",
        target="jiuding:gpu",
        image="flagquantum-runtime:v0.2.0",
        accelerator_count=2,
        cpus=8,
        memory_gib=32,
    )

    assert created["id"] == "workspace-id"
    assert created["resources"] == {
        "target": "jiuding:gpu",
        "cpus": 8,
        "memory_gib": 32,
        "gpus": 2,
        "accelerator_model": "NVIDIA_A100-SXM4-40GB",
    }
    client._catalog_image.assert_called_once_with(
        "flagquantum-runtime:v0.2.0", "PRIVATE"
    )
    path, body, _ = client._request.call_args.args
    assert path == "/api/v1/workspaces/create"
    assert body["imageId"] == "private-image"
    assert body["queueId"] == "q"
    assert body["isAutoSaveSnap"] is False
    assert body["isJupyterCloudIde"] is False
    assert body["isPrivileged"] is False
    assert body["quotaDetail"]["resourceDetail"] == {
        "acceleratorModel": "NVIDIA_A100-SXM4-40GB",
        "acceleratorCount": 2,
        "cpuCores": 8,
        "memGib": 32,
        "quotaItemId": 0,
        "sharedMemGib": 16,
    }


@pytest.mark.parametrize(
    ("target", "accelerator_count", "message"),
    [
        ("jiuding:cpu", 1, "CPU workspaces"),
        ("jiuding:gpu", 0, "GPU workspaces"),
        ("jiuding:gpu", -1, "non-negative integer"),
    ],
)
def test_create_workspace_rejects_incompatible_accelerator_count(
    client, target, accelerator_count, message
):
    client._request = Mock()
    with pytest.raises(ValueError, match=message):
        client.create_workspace(
            "flagquantum-dev",
            target=target,
            image="flagquantum-runtime:v0.2.0",
            accelerator_count=accelerator_count,
        )
    client._request.assert_not_called()


def test_workspace_lifecycle_uses_resolved_owner_and_never_saves_on_stop(client):
    client._workspace_record = Mock(
        return_value={
            "id": "workspace-id",
            "creatorId": "owner",
            "isPrivileged": False,
        }
    )
    client._request = Mock(side_effect=[{"accepted": True}, {"accepted": True}])

    assert client.start_workspace("flagquantum-dev") == {"accepted": True}
    assert client.stop_workspace("flagquantum-dev") == {"accepted": True}

    start, stop = client._request.call_args_list
    assert start.args[:2] == (
        "/api/v1/workspaces/workspace-id/restart",
        {"id": "workspace-id", "userId": "owner", "isPrivileged": False},
    )
    assert start.kwargs == {"method": "PUT"}
    assert stop.args[:2] == (
        "/api/v1/workspaces/workspace-id/stop",
        {"id": "workspace-id", "userId": "owner", "saveSnapshot": False},
    )
    assert stop.kwargs == {"method": "PUT"}


def test_uncertain_create_is_not_retried(client, tmp_path):
    script = tmp_path / "task.py"
    script.write_text("def main(): return 42")
    receipt = tmp_path / "receipt.json"
    client._request = Mock(side_effect=RuntimeError("timeout"))
    with pytest.raises(RuntimeError, match="timeout"):
        client.submit(script, target="jiuding:cpu", image="image", receipt=receipt)
    saved = json.loads(receipt.read_text())
    assert saved["submission"] == "unknown"
    assert "test-token" not in receipt.read_text()
    with pytest.raises(FileExistsError):
        client.submit(script, target="jiuding:cpu", image="image", receipt=receipt)
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


def test_status_filters_jobs_by_experiment_at_the_server(client):
    client._request = Mock(
        return_value={
            "jobInfos": [
                {
                    "id": "job",
                    "experimentId": "exp",
                    "queueId": "q",
                    "status": "Succeed",
                }
            ]
        }
    )

    jobs = client.status(
        {"endpoint": client.endpoint, "queueId": "q", "experimentId": "exp"}
    )

    assert jobs == [
        {"id": "job", "status": "Succeed", "createdTime": None, "endTime": None}
    ]
    path, body, _ = client._request.call_args.args
    assert path == "/api/v1/job/select"
    assert body["experimentId"] == "exp"


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
    receipt = client.submit(
        script,
        target="jiuding:cpu",
        image="image",
        receipt=tmp_path / "receipt.json",
    )
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


def test_first_job_resolves_exact_public_catalog_image(client, tmp_path):
    script = tmp_path / "gpu.py"
    script.write_text("def main(): return 42")
    client._jobs = Mock(return_value=[])
    config = {
        "confId": "conf",
        "configName": "config1",
        "resourceConfigList": [
            {
                "queueId": "q",
                "roleInfoList": [
                    {
                        "name": "Master",
                        "replicas": 1,
                        "resourceRequestDetail": {
                            "acceleratorModel": "NVIDIA_A100-SXM4-40GB",
                            "acceleratorCount": 1,
                            "cpuCores": 2,
                            "memGib": 2,
                            "sharedMemGib": 1,
                            "rdmaSharedCount": 0,
                        },
                    }
                ],
            }
        ],
    }
    client._request = Mock(
        side_effect=[
            {
                "items": [
                    {
                        "id": "image-id",
                        "name": "pytorch-26.03-py3-sshd",
                        "tag": "v2.0.1",
                        "status": 10,
                        "clusterId": "c",
                        "baseUrl": "harbor.example/pytorch-26.03-py3-sshd:v2.0.1",
                    }
                ]
            },
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

    receipt = client.submit(
        script,
        target="jiuding:gpu",
        image="pytorch-26.03-py3-sshd:v2.0.1",
        receipt=tmp_path / "receipt.json",
    )

    assert receipt["jobId"] == "job"
    client._jobs.assert_not_called()
    calls = client._request.call_args_list
    assert calls[0].args[0] == "/api/v1/images/select"
    resource = calls[1].args[1]["advanceConfigInfos"][0]["resourceConfigList"][0]
    assert resource["basicImage"] == ("harbor.example/pytorch-26.03-py3-sshd:v2.0.1")
    assert resource["imageRegion"] == "PUBLIC"


def test_submit_resolves_exact_private_catalog_image(client, tmp_path):
    script = tmp_path / "gpu.py"
    script.write_text("def main(): return 42")
    client._catalog_image = Mock(
        return_value={
            "id": "private-image-id",
            "name": "flagquantum-runtime",
            "tag": "v0.2.0",
            "registryUrl": "harbor.internal/flagquantum-runtime:v0.2.0",
        }
    )
    client._request = Mock(
        side_effect=[
            {"experimentId": "exp"},
            {
                "experimentSummaryInfos": [
                    {
                        "experimentId": "exp",
                        "advanceConfigInfos": [
                            {
                                "confId": "conf",
                                "configName": "config1",
                                "resourceConfigList": [
                                    {
                                        "queueId": "q",
                                        "roleInfoList": [
                                            {
                                                "name": "Master",
                                                "replicas": 1,
                                                "resourceRequestDetail": {
                                                    "acceleratorModel": "NVIDIA_A100-SXM4-40GB",
                                                    "acceleratorCount": 1,
                                                    "cpuCores": 2,
                                                    "memGib": 2,
                                                    "sharedMemGib": 1,
                                                    "rdmaSharedCount": 0,
                                                },
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                        "creatorId": "u",
                        "nativeCluster": "default-cluster",
                    }
                ]
            },
            {"jobId": "job"},
        ]
    )

    receipt = client.submit(
        script,
        target="jiuding:gpu/NVIDIA_A100-SXM4-40GB",
        image="flagquantum-runtime:v0.2.0",
        image_region="PRIVATE",
        receipt=tmp_path / "receipt.json",
    )

    assert receipt["jobId"] == "job"
    client._catalog_image.assert_called_once_with(
        "flagquantum-runtime:v0.2.0", "PRIVATE"
    )
    resource = client._request.call_args_list[0].args[1]["advanceConfigInfos"][0][
        "resourceConfigList"
    ][0]
    assert resource["basicImage"] == ("harbor.internal/flagquantum-runtime:v0.2.0")
    assert resource["imageRegion"] == "PRIVATE"


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
        client.submit(script, target="jiuding:cpu", image="image", receipt=path)
    saved = json.loads(path.read_text())
    assert saved["experimentId"] == "exp"
    assert saved["submission"] == "launch_unknown"
    with pytest.raises(FileExistsError):
        client.submit(script, target="jiuding:cpu", image="image", receipt=path)
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


@pytest.mark.parametrize("chip_type", ["mlu", "npu", "xpu"])
def test_reserved_targets_fail_before_network(client, tmp_path, chip_type):
    client._request = Mock()
    with pytest.raises(NotImplementedError, match="reserved"):
        client.submit(
            tmp_path / "missing.py",
            target=f"jiuding:{chip_type}",
            image="image",
            receipt=tmp_path / "r.json",
        )
    client._request.assert_not_called()


@pytest.mark.parametrize(
    "target",
    ["gpu", "other:gpu", "jiuding:tpu", "jiuding:gpu/", "jiuding:cpu/model"],
)
def test_invalid_compute_targets_fail_before_network(client, tmp_path, target):
    client._request = Mock()
    with pytest.raises((ValueError, NotImplementedError)):
        client.submit(
            tmp_path / "missing.py",
            target=target,
            image="image",
            receipt=tmp_path / "r.json",
        )
    client._request.assert_not_called()


@pytest.mark.parametrize("mismatch", [False, True])
def test_gpu_submit_validates_saved_resources_before_launch(client, tmp_path, mismatch):
    script = tmp_path / "gpu.py"
    script.write_text("def main(): return 42")
    client._jobs = Mock(
        return_value=[
            {
                "queueId": "q",
                "basicImage": "image",
                "imageRegion": "PUBLIC",
                "roleInfos": [
                    {
                        "resourceRequestDetail": {
                            "acceleratorModel": "NVIDIA_A100-SXM4-40GB"
                        }
                    }
                ],
            }
        ]
    )
    saved = {}

    def request(path, body, headers):
        if path == "/api/v1/experiment":
            saved.update(body["advanceConfigInfos"][0])
            saved["confId"] = "conf"
            if mismatch:
                saved["resourceConfigList"][0]["roleInfoList"][0][
                    "resourceRequestDetail"
                ]["acceleratorCount"] = 0
            return {"experimentId": "exp"}
        if path == "/api/v1/experiment/select":
            return {
                "experimentSummaryInfos": [
                    {
                        "experimentId": "exp",
                        "advanceConfigInfos": [saved],
                        "creatorId": "u",
                        "nativeCluster": "default-cluster",
                    }
                ]
            }
        return {"jobId": "job"}

    client._request = Mock(side_effect=request)
    receipt = tmp_path / "r.json"
    if mismatch:
        with pytest.raises(RuntimeError, match="differs"):
            client.submit(script, target="jiuding:gpu", image="image", receipt=receipt)
        assert client._request.call_count == 2
        assert json.loads(receipt.read_text())["submission"] == "created"
    else:
        record = client.submit(
            script, target="jiuding:gpu", image="image", receipt=receipt
        )
        assert record["resources"]["gpus"] == 1
        assert saved["resourceConfigList"][0]["roleInfoList"][0]["replicas"] == 1
        assert "--gpus 1" in saved["command"]
        assert client._request.call_count == 3


def test_gpu_target_must_match_model_observed_for_image(client, tmp_path):
    script = tmp_path / "gpu.py"
    script.write_text("def main(): return 42")
    client._jobs = Mock(
        return_value=[
            {
                "queueId": "q",
                "basicImage": "image",
                "imageRegion": "PUBLIC",
                "roleInfos": [{"resourceRequestDetail": {"acceleratorModel": model}}],
            }
            for model in ("NVIDIA_A100", "NVIDIA_H100")
        ]
    )
    client._request = Mock()
    with pytest.raises(ValueError, match="target model"):
        client.submit(
            script,
            target="jiuding:gpu",
            image="image",
            receipt=tmp_path / "r.json",
        )
    client._request.assert_not_called()


@pytest.mark.parametrize("available,count", [(False, 0), (True, 2)])
def test_gpu_worker_fails_before_user_code_without_one_gpu(
    tmp_path, monkeypatch, available, count
):
    from flagquantum.compute.registry import get_platform_runtime
    from flagquantum.remote.compute import _worker

    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["worker", str(tmp_path / "missing.py"), str(output), "run", "--gpus", "1"],
    )
    monkeypatch.setattr(
        get_platform_runtime("cuda"),
        "discover",
        lambda: (None,) * count if available else (),
    )
    with pytest.raises(RuntimeError, match="exactly one CUDA"):
        _worker.main()
    assert not output.exists()
