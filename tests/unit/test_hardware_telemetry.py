"""Observe benchmark conditions without changing workload or watchdog verdicts."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import hardware_telemetry as telemetry
from tools import run_with_watchdog as watchdog

pytestmark = pytest.mark.unit


def test_samples_retain_clocks_load_and_owned_affinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def probe(
        command: tuple[str, ...], *, text: bool, capture_output: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert text and capture_output
        assert timeout == (0.5 if command[0] == "nvidia-smi" else 0.25)
        output = (
            "0, P0, 1200, 900, 50, 10, 1024, 150.5, 60\n"
            if command[0] == "nvidia-smi"
            else "7 19 3 5.0 128\n9 44 1 6.0 256\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    seen = []

    def affinity(pid: int) -> set[int]:
        seen.append(pid)
        return {3, 2}

    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(os, "sched_getaffinity", affinity, raising=False)
    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 2.0, 3.0))
    output = tmp_path / "samples.jsonl"
    for _ in range(2):
        telemetry.record_sample(output, 19, time.monotonic() - 1)
    samples = [json.loads(row) for row in output.read_text().splitlines()]
    assert len(samples) == 2 and seen == [7, 7]
    for sample in samples:
        assert sample["process_group"] == 19
        assert sample["elapsed_seconds"] >= 1
        assert sample["collection_seconds"] >= 0
        observations = sample["observations"]
        assert observations["gpu"] == "0, P0, 1200, 900, 50, 10, 1024, 150.5, 60"
        assert observations["owned_process_affinities"] == "pid=7 allowed_cpus=[2, 3]"
        assert observations["load_average_1_5_15_minutes"] == "(1.0, 2.0, 3.0)"
        assert observations["processes"] == "7 19 3 5.0 128"
        assert "cuda_visible_devices" not in observations


@pytest.mark.parametrize("failure", ["timeout", "missing", "nonzero", "decode"])
def test_unavailable_probes_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def probe(
        command: tuple[str, ...], *, text: bool, capture_output: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, timeout, output="GPU-private", stderr="private-host"
            )
        if failure == "decode":
            raise UnicodeError("private-host")
        if failure == "missing":
            raise FileNotFoundError("private-host")
        return subprocess.CompletedProcess(command, 3, "GPU-private", "private-host")

    monkeypatch.setattr(subprocess, "run", probe)
    output = tmp_path / "samples.jsonl"
    telemetry.record_sample(output, 19, time.monotonic())
    assert "GPU-private" not in output.read_text()
    assert "private-host" not in output.read_text()
    observations = json.loads(output.read_text())["observations"]
    if failure == "nonzero":
        assert observations["gpu_error"] == "exit_status=3"
        assert "gpu" not in observations
    else:
        assert "gpu_error" in observations and "processes_error" in observations
        assert "gpu" not in observations


def test_samples_exclude_identifiers_and_unrelated_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def probe(
        command: tuple[str, ...], *, text: bool, capture_output: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            assert "uuid" not in " ".join(command)
            output = "GPU-private, P0, 1200, 900, 50, 10, 1024, 150, 60\n"
        else:
            assert "comm=" not in " ".join(command)
            output = "987654 44 1 6.0 256\n7 19 3 5.0 128\nprivate-host\n"
        return subprocess.CompletedProcess(command, 0, output, "GPU-private")

    def denied(pid: int) -> set[int]:
        assert pid == 7
        raise OSError("private-host")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-private")
    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(os, "sched_getaffinity", denied, raising=False)
    output = tmp_path / "samples.jsonl"
    telemetry.record_sample(output, 19, time.monotonic())
    encoded = output.read_text()
    assert all(
        value not in encoded for value in ("GPU-private", "private-host", "987654")
    )
    observations = json.loads(encoded)["observations"]
    assert observations["gpu_error"] == "unrecognized GPU accounting row"
    assert observations["processes_error"] == "unrecognized process accounting row"
    assert observations["gpu_diagnostic"] == "stderr_present"
    assert observations["owned_process_affinities"] == "pid=7 unavailable: OSError"


@pytest.mark.parametrize("gpu", ["", "0, N/A, N/A, [Not Supported], 0, 0, 0, 0, 0\n"])
def test_unavailable_metrics_and_load_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: str
) -> None:
    def probe(
        command: tuple[str, ...], *, text: bool, capture_output: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 0, gpu if command[0] == "nvidia-smi" else "", ""
        )

    def denied() -> tuple[float, float, float]:
        raise OSError("private-host")

    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(os, "getloadavg", denied)
    output = tmp_path / "samples.jsonl"
    telemetry.record_sample(output, 19, time.monotonic())
    observations = json.loads(output.read_text())["observations"]
    assert observations["load_average_error"] == "OSError"
    assert "private-host" not in output.read_text()
    if gpu:
        assert observations["gpu"] == gpu.strip()
        assert "gpu_error" not in observations
    else:
        assert observations["gpu_error"] == "empty GPU accounting"


def _arguments(tmp_path: Path, code: str) -> list[str]:
    return [
        "--phase",
        "telemetry-test",
        "--stall-seconds",
        "2",
        "--timeout-seconds",
        "3",
        "--diagnostics",
        str(tmp_path / "diagnostics"),
        "--hardware-samples",
        str(tmp_path / "samples.jsonl"),
        "--",
        sys.executable,
        "-c",
        code,
    ]


def test_watchdog_samples_through_command_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups = []

    def record(destination: Path, group: int, started: float) -> None:
        groups.append(group)
        assert time.monotonic() >= started
        with destination.open("a") as stream:
            stream.write("observed\n")

    monkeypatch.setattr(telemetry, "record_sample", record)
    result = watchdog.run(
        _arguments(tmp_path, "import time,sys; time.sleep(1.2); sys.exit(7)")
    )
    assert result == 7
    assert len(groups) >= 2 and len(set(groups)) == 1
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == len(groups)


def test_sampling_preserves_watchdog_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry, "record_sample", lambda *args: None)
    arguments = _arguments(tmp_path, "import time; time.sleep(5)")
    arguments[arguments.index("--timeout-seconds") + 1] = "0.2"
    assert watchdog.run(arguments) == 124
    result = json.loads((tmp_path / "diagnostics/watchdog.json").read_text())
    assert result["reason"] == "phase_timeout"
    assert result["cleanup_required"] is True


def test_existing_observations_are_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "samples.jsonl"
    output.write_text("retained\n")
    with pytest.raises(FileExistsError):
        watchdog.run(_arguments(tmp_path, "raise RuntimeError('must not launch')"))
    assert output.read_text() == "retained\n"


def test_strict_rss_cadence_cannot_silently_degrade(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, "raise RuntimeError('must not launch')")
    arguments[0:0] = ["--rss-limit-bytes", "1024"]
    with pytest.raises(SystemExit) as error:
        watchdog.run(arguments)
    assert error.value.code == 2
    assert not (tmp_path / "samples.jsonl").exists()


def test_recording_failure_terminates_the_owned_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(destination: Path, group: int, started: float) -> None:
        raise OSError("observation write failed")

    monkeypatch.setattr(telemetry, "record_sample", unavailable)
    marker = tmp_path / "finished"
    code = f"import time; time.sleep(.5); open({str(marker)!r}, 'w').close()"
    with pytest.raises(OSError, match="observation write failed"):
        watchdog.run(_arguments(tmp_path, code))
    time.sleep(0.6)
    assert not marker.exists()
    result = json.loads((tmp_path / "diagnostics/watchdog.json").read_text())
    assert result["reason"] == "supervisor_interrupted"
    assert result["cleanup_required"] is True


def test_observed_workflows_retain_scientific_work_and_deadlines() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/local-gpu.yml"
    ).read_text()
    for phase, loops, timeout in (
        ("one-gpu-benchmark", "20", "900"),
        ("distributed-benchmark", "2000", "1200"),
    ):
        lines = [line for line in workflow.splitlines() if f"--phase {phase} " in line]
        assert len(lines) == 1
        command = shlex.split(lines[0])
        assert command[command.index("--timeout-seconds") + 1] == timeout
        assert command.count("--hardware-samples") == 1
        workload = command[command.index("--") + 1 :]
        assert workload[workload.index("--warmup") + 1] == "10"
        assert workload[workload.index("--repetitions") + 1] == "30"
        assert workload[workload.index("--inner-loops") + 1] == loops


def test_direct_script_runs_without_repository_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    script = Path(watchdog.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *_arguments(tmp_path, "import time; time.sleep(.4)"),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    samples = (tmp_path / "samples.jsonl").read_text().splitlines()
    assert samples and json.loads(samples[0])["process_group"] > 0


@pytest.mark.parametrize("group", [None, 19])
def test_failure_diagnostics_exclude_unowned_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group: int | None
) -> None:
    def probe(
        command: tuple[str, ...], *, text: bool, capture_output: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert group is not None
        assert "uuid" not in " ".join(command) and "cmd" not in " ".join(command)
        output = (
            "GPU-private, P0, 1200, 900, 50, 10, 1024, 150, 60\n"
            if command[0] == "nvidia-smi"
            else "987654 44 1 6.0 256\n7 19 3 5.0 128\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "private-host")

    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2}, raising=False)
    watchdog._diagnostics(tmp_path, {"reason": "phase_timeout"}, group=group)
    for output in tmp_path.iterdir():
        assert all(
            value not in output.read_text()
            for value in ("GPU-private", "private-host", "987654")
        )
    assert (
        json.loads((tmp_path / "watchdog.json").read_text())["reason"]
        == "phase_timeout"
    )
    processes = json.loads((tmp_path / "processes.log").read_text())
    if group is None:
        assert processes["snapshot_error"] == "command_not_started"
    else:
        assert processes["processes"] == "7 19 3 5.0 128"
