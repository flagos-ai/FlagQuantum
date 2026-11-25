"""Native jobs carry circuits and checked results without a workspace transport."""

import base64
import hashlib
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.remote.compute._native_job import (
    NativeJiudingJobClient,
    _command,
    _decode_lines,
)
from flagquantum.remote.compute._workspace_executor import SCHEMA, VERSION

pytestmark = pytest.mark.unit


def test_inline_command_executes_circuit_and_decodes_result():
    ir = fq.Circuit(2).x(0).to_ir()
    command = _command(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "operation": "statevector",
            "request_id": "batch-program",
            "target": "jiuding:cpu",
            "program": ir.to_dict(),
        },
        "test-run",
    )
    args = shlex.split(command)
    args[0] = sys.executable
    output = subprocess.run(
        args, capture_output=True, text=True, timeout=45, check=True
    )
    result = _decode_lines(output.stdout.splitlines(), "test-run")
    assert result["ok"] is True
    from flagquantum.remote.compute._program_job import decode_program_result

    decoded = decode_program_result(
        result, requested_target="jiuding:cpu", selected_target="jiuding:cpu"
    )
    expected = fq.run(fq.Circuit(2).x(0))
    import torch

    torch.testing.assert_close(decoded.state, expected.state)


def test_multi_device_command_uses_one_torchrun_rank_per_device():
    command = _command(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "operation": "statevector",
            "request_id": "batch-program",
            "target": "jiuding:gpu/A100",
            "program": fq.Circuit(2).to_ir().to_dict(),
        },
        "test-run",
        devices=4,
    )
    args = shlex.split(command)
    assert args[:5] == [
        "python",
        "-u",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
    ]
    assert "--master-addr=127.0.0.1" in args
    assert "--master-port=29500" in args
    assert "--nproc-per-node=4" in args
    assert args[-3] == "flagquantum.remote.compute._distributed_job"


def chunks(value):
    data = json.dumps(value).encode()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode()
    parts = [encoded[i : i + 100] for i in range(0, len(encoded), 100)]
    return [
        f"FQ_RESULT_run {i}/{len(parts)} {digest} {part}"
        for i, part in enumerate(parts)
    ]


def test_result_chunks_reorder_deduplicate_and_ignore_other_runs():
    value = {"message": "large result" * 50}
    lines = chunks(value)
    assert _decode_lines(["other output", *reversed(lines), lines[0]], "run") == value
    with pytest.raises(RuntimeError, match="no result yet"):
        _decode_lines(lines[:-1], "run")
    with pytest.raises(RuntimeError, match="Conflicting"):
        _decode_lines([*lines, lines[0] + "A"], "run")
    damaged = lines.copy()
    fields = damaged[0].split(" ")
    fields[-1] = "A" + fields[-1][1:]
    damaged[0] = " ".join(fields)
    with pytest.raises(RuntimeError, match="digest"):
        _decode_lines(damaged, "run")


def test_transport_bounds_precede_execution():
    with pytest.raises(ValueError, match="64 KiB"):
        _command({"payload": "x" * 65536}, "run")
    with pytest.raises(RuntimeError, match="bounds"):
        _decode_lines(["FQ_RESULT_run 0/4097 digest a"], "run")
    with pytest.raises(RuntimeError, match="Malformed"):
        _decode_lines(["FQ_RESULT_run invalid"], "run")


@pytest.mark.parametrize("devices", (0, -1, 1.5, True))
def test_native_submission_rejects_invalid_device_count_before_context(devices):
    client = NativeJiudingJobClient(project="set.project", queue="queue")
    with pytest.raises(ValueError, match="devices must be a positive integer"):
        client.submit_program(
            fq.Circuit(2),
            target="jiuding:gpu/A100",
            image="image",
            devices=devices,
        )


def test_project_context_uses_authenticated_user_and_queue(monkeypatch):
    client = NativeJiudingJobClient(project="set.project", queue="queue")
    monkeypatch.setattr(client, "_auth", lambda: {"AIRS-Token": "private"})
    paths = []

    def request(path, body, headers, **kwargs):
        paths.append(path)
        if path.endswith("userinfo"):
            return {"data": {"id": "own-user", "alias": "user"}}
        if path == "/api/v1/projsets/select-joined":
            return {"items": [{"projsetInfo": {"name": "set", "id": "set-id"}}]}
        if path == "/api/v1/projects/select-joined":
            assert body == {"projsetId": "set-id"}
            return {"items": [{"projInfo": {"name": "project", "id": "project-id"}}]}
        raise AssertionError(path)

    def pages(path, body, key, headers):
        assert path == "/api/v1/queue/select"
        assert body["as_user"] is True and body["userId"] == "own-user"
        assert headers["x-user-id"] == "own-user"
        return [
            {
                "name": "queue",
                "id": "queue-id",
                "status": "QUEUE_STATUS_ACTIVE",
                "quotaInfoList": [
                    {
                        "clusterId": "cluster",
                        "clusterName": "cluster",
                        "zoneQuotaInfoList": [
                            {
                                "zoneId": "zone",
                                "quotaMetaList": [
                                    {
                                        "priority": "high",
                                        "resourceDetail": {"acceleratorModel": "A100"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "_pages", pages)
    context = client.workspace()
    assert context["queueId"] == "queue-id"
    assert context["storageInfo"] == []
    assert all("workspaces" not in path for path in paths)
    assert client.workspace() == context
    assert len(paths) == 3


def test_multi_device_submission_reserves_the_same_number_of_accelerators(
    monkeypatch,
):
    client = NativeJiudingJobClient(project="set.project", queue="queue")
    monkeypatch.setattr(
        client,
        "workspace",
        lambda: {
            "queueId": "queue-id",
            "queueName": "queue",
            "clusterName": "cluster",
            "acceleratorModel": "A100",
        },
    )
    monkeypatch.setattr(
        client,
        "_catalog_image",
        lambda image, region: {
            "registryUrl": "registry/image",
            "imageRegion": "PROJSET",
        },
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: pytest.fail("Submission must not query Grafana"),
    )
    monkeypatch.setattr(client, "_auth", lambda: {"AIRS-Token": "token"})
    captured = {}

    def launch(**options):
        captured.update(options)
        return {"jobId": "job-id", **options["record"]}

    monkeypatch.setattr(client, "_launch_command", launch)
    receipt = client.submit_program(
        fq.Circuit(3).h(0).cx(0, 2),
        target="jiuding:gpu/A100",
        image="image",
        devices=4,
    )
    assert captured["gpus"] == 4
    assert captured["image_region"] == "PROJSET"
    assert captured["cpus"] >= 4
    assert "--nproc-per-node=4" in captured["command"]
    assert receipt["devices"] == 4
    assert receipt["cluster_name"] == "cluster"
    assert "datasource_uid" not in receipt
    assert receipt["execution_distribution_semantics"] == "sharded_across_ranks"


@pytest.fixture
def telemetry_job(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[NativeJiudingJobClient, dict[str, Any]]:
    client = NativeJiudingJobClient(project="set.project", queue="queue")
    monkeypatch.setattr(client, "_auth", lambda: {"AIRS-Token": "token"})
    monkeypatch.setattr(
        client, "workspace", lambda: {"queueId": "queue-id", "clusterName": "cluster"}
    )
    return client, {
        "project": "set.project",
        "endpoint": client.endpoint,
        "queueId": "queue-id",
        "jobId": "00000000-0000-0000-0000-000000000001",
        "run_id": "0" * 32,
        "created_ms": 1_000_000,
        "requested_target": "jiuding:cpu",
        "selected_target": "jiuding:cpu",
        "datasource_uid": "retired-grafana-source",
    }


@pytest.mark.parametrize("camel_case", [False, True])
def test_telemetry_pages_restore_old_receipt_and_decode_state(
    telemetry_job: tuple[NativeJiudingJobClient, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    camel_case: bool,
) -> None:
    import torch

    from flagquantum.remote.compute._workspace_executor import execute

    client, receipt = telemetry_job
    circuit = fq.Circuit(3).x(0)
    value = execute(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "operation": "statevector",
            "request_id": "batch-program",
            "target": "jiuding:cpu",
            "program": circuit.to_ir().to_dict(),
        },
        target="jiuding:cpu",
        device="cpu",
    )
    lines = [
        line.replace("FQ_RESULT_run ", f"FQ_RESULT_{receipt['run_id']} ")
        for line in chunks(value)
    ]
    pages = [lines[::2], [*lines[1::2], lines[0]]]
    calls = []
    monkeypatch.setattr(
        "flagquantum.remote.compute._native_job.time.time", lambda: 2000
    )

    def request(
        path: str, body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        calls.append((path, deepcopy(body), headers))
        return {
            "code": 0,
            "data": {
                "entries": [{"message": line} for line in pages[len(calls) - 1]],
                "hasMore" if camel_case else "has_more": len(calls) == 1,
                "nextCursor" if camel_case else "next_cursor": "opaque-cursor",
            },
        }

    monkeypatch.setattr(client, "_request", request)
    result = client.read_result(receipt)
    torch.testing.assert_close(result.state, fq.run(circuit).state)
    assert len(calls) == 2
    path, first, headers = calls[0]
    assert path == "/api/v1/telemetry/logs/query"
    assert headers == {"AIRS-Token": "token"}
    assert first == {
        "workload_id": receipt["jobId"],
        "workload_type": 1,
        "cluster_name": "cluster",
        "time_range": {"from": 940_000, "to": 2_000_000},
        "filter_options": {
            "log_source": 1,
            "keyword": f"FQ_RESULT_{receipt['run_id']}",
        },
        "pagination": {"limit": 4096, "direction": 2},
    }
    assert calls[1][1] == {
        **first,
        "pagination": {**first["pagination"], "cursor": "opaque-cursor"},
    }


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"code": 50001, "data": None}, "code 50001"),
        ({"code": 0, "data": None}, "Malformed"),
        ({"code": 0, "data": {"entries": [{}]}}, "Malformed"),
        ({"code": 0, "data": {"entries": [{"message": "x" * 4097}]}}, "bounds"),
        ({"code": 0, "data": {"entries": [], "has_more": True}}, "pagination"),
        ({"code": 0, "data": {"entries": [], "has_more": False}}, "has no result yet"),
    ],
)
def test_telemetry_failures_are_not_empty_successes(
    telemetry_job: tuple[NativeJiudingJobClient, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    error: str,
) -> None:
    client, receipt = telemetry_job
    monkeypatch.setattr(client, "_request", lambda *args: response)
    with pytest.raises(RuntimeError, match=error):
        client.read_result(receipt)


def test_telemetry_clamps_old_window_and_stops_repeated_cursor(
    telemetry_job: tuple[NativeJiudingJobClient, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, receipt = telemetry_job
    receipt["cluster_name"] = "submitted-cluster"
    monkeypatch.setattr(
        "flagquantum.remote.compute._native_job.time.time", lambda: 4_000_000
    )
    calls = []

    def request(
        path: str, body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        calls.append(deepcopy(body))
        return {
            "code": 0,
            "data": {"entries": [], "has_more": True, "next_cursor": "same"},
        }

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError, match="pagination did not advance"):
        client.read_result(receipt)
    assert len(calls) == 2
    assert calls[0]["cluster_name"] == "submitted-cluster"
    assert calls[0]["time_range"] == {
        "from": 4_000_000_000 - 30 * 86400_000,
        "to": 4_000_000_000,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project", "other.project"),
        ("endpoint", "https://other"),
        ("queueId", "other-queue"),
    ],
)
def test_telemetry_checks_receipt_ownership_before_query(
    telemetry_job: tuple[NativeJiudingJobClient, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    client, receipt = telemetry_job
    receipt[field] = value
    monkeypatch.setattr(
        client, "_request", lambda *args: pytest.fail("Invalid receipt queried logs")
    )
    with pytest.raises(ValueError, match="belongs to another"):
        client.read_result(receipt)
