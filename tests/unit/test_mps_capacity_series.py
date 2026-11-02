"""A capacity campaign varies deployment and terminates only its own workers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import run_mps_capacity_series as series

pytestmark = pytest.mark.unit


def test_reserved_pair_restores_exact_source_and_reruns_the_same_artifact(
    tmp_path: Path,
) -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/scheduled-hardware.yml"
    )
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    producer = jobs["prepare-mps-source"]
    consumer = jobs["mps-capacity-series"]
    assert "flagquantum-a100-cpu-entry" not in producer["runs-on"]
    assert "'flagquantum-a100-cpu'" in producer["runs-on"]
    assert "'ubuntu-latest'" in producer["runs-on"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in producer["runs-on"]
    )
    assert (
        producer["outputs"]["artifact-key"]
        == "${{ steps.pack-source.outputs.artifact-key }}"
    )
    download = next(
        s
        for s in consumer["steps"]
        if s.get("uses", "").startswith("actions/download-artifact@")
    )
    assert (
        "${{ needs.prepare-mps-source.outputs.artifact-key }}"
        in download["with"]["name"]
    )
    pack = next(s["run"] for s in producer["steps"] if s.get("id") == "pack-source")
    restore = next(
        s["run"]
        for s in consumer["steps"]
        if s.get("name") == "Restore and verify this run's exact source commit"
    )
    source, target, temporary = (
        tmp_path / name for name in ("source", "target", "temporary")
    )
    for directory in (source, target, temporary):
        directory.mkdir()
    env = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        RUNNER_TEMP=str(temporary),
        GITHUB_RUN_ID="123",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_OUTPUT=str(temporary / "output"),
    )

    def run(
        command: list[str], cwd: Path, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=True,
            text=True,
            timeout=20,
        )

    for args in (
        ("init",),
        ("config", "user.name", "FlagQuantum"),
        ("config", "user.email", "liuwei.chem.phys@gmail.com"),
    ):
        run(["git", *args], source)
    (source / "workload.txt").write_text("frozen circuit\n")
    (source / "binary.dat").write_bytes(bytes(range(256)))
    run(["git", "add", "."], source)
    run(["git", "commit", "-m", "frozen"], source)
    sha = run(["git", "rev-parse", "HEAD"], source).stdout.strip()
    env["GITHUB_SHA"] = sha
    run(["bash", "-eo", "pipefail", "-c", pack], source)
    assert (temporary / "output").read_text().strip() == "artifact-key=123-1"
    env["GITHUB_RUN_ATTEMPT"] = "2"
    for _ in range(2):
        run(["bash", "-eo", "pipefail", "-c", restore], target)
        restored = target / "capacity-series-source"
        assert run(["git", "rev-parse", "HEAD"], restored).stdout.strip() == sha
        assert (restored / "binary.dat").read_bytes() == bytes(range(256))
        assert (restored / "workload.txt").read_text() == "frozen circuit\n"
        (restored / "workload.txt").write_text("dirty\n")
    (temporary / "mps-source/source.sha").write_text("0" * 40 + "\n")
    assert (
        run(["bash", "-eo", "pipefail", "-c", restore], target, False).returncode != 0
    )


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
