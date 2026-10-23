"""A capacity campaign varies deployment and terminates only its own workers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_mps_capacity_series as series

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
def test_series_commands_keep_the_same_frozen_manifest(world: int) -> None:
    source, output = Path("/nfs/fq-mps-capacity-test"), Path("/nfs/test-out")
    nodes = 2 if world == 16 else 1
    for node in range(nodes):
        command = series.point_command(
            world, node, source, output, "10.1.15.172", 29640
        )
        assert command[command.index("--nproc-per-node") + 1] == str(min(world, 8))
        assert command[command.index("--nnodes") + 1] == str(nodes)
        assert command[command.index("--release-manifest") + 1] == str(
            source / "benchmarks/manifests/mps_release_v1.json"
        )
        assert command[command.index("--role") + 1] == "capacity-series-point"
        assert "--bond-dimension" not in command


def test_transport_is_checked_at_the_full_topology_before_capacity() -> None:
    command = series.point_command(
        16,
        1,
        Path("/nfs/fq-mps-capacity-test"),
        Path("/nfs/test-out"),
        "10.1.15.172",
        29640,
        transport_check=True,
    )
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[-2:] == ["--backend", "nccl"]
    assert command[-3].endswith("tests/distributed/mps_training_runtime.py")
    assert "--role" not in command


def test_cleanup_refuses_a_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text(json.dumps({"pid": 12345, "start_time": "older"}))
    monkeypatch.setattr(series, "_start_time", lambda pid: "newer")

    def must_not_signal(pid: int, sig: int) -> None:
        pytest.fail("cleanup must not signal a process whose start time changed")

    monkeypatch.setattr(series.os, "killpg", must_not_signal)
    with pytest.raises(RuntimeError, match="identity changed"):
        series.stop_worker(pid_file)


def test_cleanup_of_an_exited_worker_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text(json.dumps({"pid": 12345, "start_time": "old"}))

    def exited(pid: int) -> str:
        raise FileNotFoundError

    monkeypatch.setattr(series, "_start_time", exited)
    series.stop_worker(pid_file)
    series.stop_worker(pid_file)
    assert not pid_file.exists()
