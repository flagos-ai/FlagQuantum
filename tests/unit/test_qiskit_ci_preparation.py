"""Qiskit caches reuse bytes without hiding a missing version or dependency."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_qiskit_gate_requires_both_complete_certified_lanes(result: str) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    lane = jobs["qiskit-compat"]
    assert lane["strategy"] == {
        "fail-fast": False,
        "matrix": {"qiskit": ["2.0", "2.5"]},
    }
    assert lane["env"]["QISKIT_VERSION"] == "${{ matrix.qiskit }}.*"
    commands = next(
        s
        for s in lane["steps"]
        if s.get("name") == "Verify this certified Qiskit and Aer pair"
    )
    assert commands["env"]["PYTEST_ADDOPTS"] == (
        "-n ${{ env.FQ_PYTEST_WORKERS }} --dist=loadfile --durations=50"
    )
    assert (
        "tests/benchmark_contract/test_compiler_*.py -m benchmark_contract -q"
        in commands["run"]
    )
    gate = jobs["qiskit-optional"]
    assert gate["if"] == "${{ always() }}"
    assert gate["needs"] == "qiskit-compat"
    command = gate["steps"][0]["run"].replace(
        "${{ needs.qiskit-compat.result }}", result
    )
    completed = subprocess.run(["bash", "-eu", "-c", command], timeout=10, check=False)
    assert (completed.returncode == 0) == (result == "success")


@pytest.mark.parametrize("mirror_fails", [False, True])
def test_cached_wheels_still_resolve_live_versions_and_validate_offline_closure(
    tmp_path: Path, mirror_fails: bool
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/prepare-qiskit-wheels/action.yml").read_text()
    )
    executable = tmp_path / "bin"
    executable.mkdir()
    if shutil.which("timeout") is None:
        timer = executable / "timeout"
        timer.write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
        timer.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    # Exercise the real shell, recording pip commands without using a network.
    python = executable / "python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if os.environ['MIRROR_FAILS'] == '1' and 'wheel' in sys.argv "
        "and 'https://mirror.example/simple' in sys.argv else 0)\n"
    )
    python.chmod(0o755)
    wheels = tmp_path / "private-wheels"
    wheels.mkdir()
    (wheels / "old-cached-wheel.whl").write_bytes(b"cached bytes")
    public = tmp_path / "public-wheels"
    public.mkdir()
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", action["runs"]["steps"][0]["run"]],
        env={
            **os.environ,
            "PATH": f"{executable}:{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "MIRROR_FAILS": str(int(mirror_fails)),
            "RUNNER_ENVIRONMENT": "self-hosted",
            "FQ_QISKIT_WHEEL_DIRECTORY": str(wheels),
            "FQ_QISKIT_PUBLIC_WHEELS": str(public),
            "QISKIT_VERSION": "2.5.*",
            "PIP_INDEX_URL": "https://mirror.example/simple",
        },
        check=True,
        timeout=15,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == (3 if mirror_fails else 2)
    for call in calls[:-1]:
        assert call[:3] == ["-m", "pip", "wheel"]
        assert "--index-url" in call and "--no-index" not in call
        assert str(wheels) in call and str(public) in call
        assert "qiskit[qasm3-import]==2.5.*" in call
        assert "qiskit-aer==0.17.*" in call and "sympy>=1.13,<2" in call
    if mirror_fails:
        assert "https://pypi.org/simple" in calls[1]
    assert calls[-1][:3] == ["-m", "pip", "install"]
    assert {"--dry-run", "--ignore-installed", "--no-index"} <= set(calls[-1])


def test_hosted_wheel_cache_does_not_restore_installed_environments() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    steps = jobs["qiskit-compat"]["steps"]
    cache = next(s for s in steps if s.get("uses") == "actions/cache@v4")
    assert cache["if"] == "runner.environment == 'github-hosted'"
    assert cache["with"]["path"] == "${{ env.FQ_QISKIT_WHEEL_DIRECTORY }}"
    for component in (
        "runner.os",
        "runner.arch",
        "py312",
        "matrix.qiskit",
        "dependency-policy.toml",
    ):
        assert component in cache["with"]["key"]
    assert "restore-keys" not in cache["with"]
