"""The post-merge smoke tests the installed wheel outside its source checkout."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_main_smoke_checks_installation_outside_the_checkout(tmp_path: Path) -> None:
    document = yaml.safe_load((ROOT / ".github/workflows/main-smoke.yml").read_text())
    events = document.get("on", document.get(True))
    assert events["push"] == {"branches": ["main"]}
    assert "merge_group" not in events
    steps = document["jobs"]["installed-wheel-smoke"]["steps"]
    build = next(
        step["run"]
        for step in steps
        if step.get("name") == "Build and inspect the wheel"
    )
    assert "python -m build --wheel --no-isolation" in build
    assert "verify_distribution_artifacts.py dist" in build
    command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Check the installed package outside the source checkout"
    )
    tools = tmp_path / "bin"
    tools.mkdir()
    python = tools / "python"
    python.write_text(
        '#!/bin/sh\nprintf "%s|%s\\n" "$PWD" "$*" >> "$CALLS"\n'
        'if [ "$1" = "-" ]; then cat > "$PAYLOAD"; fi\n'
    )
    python.chmod(0o755)
    calls = tmp_path / "calls"
    payload = tmp_path / "payload"
    completed = subprocess.run(
        ["bash", "-eu", "-c", command],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(tmp_path),
            "CALLS": str(calls),
            "PAYLOAD": str(payload),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recorded = calls.read_text().splitlines()
    assert len(recorded) == 2
    for call in recorded:
        assert (
            Path(call.split("|", 1)[0]).resolve()
            == (tmp_path / "main-smoke-tests").resolve()
        )
    assert recorded[-1].endswith("|-m pytest test_smoke_baseline.py -q")
    assert "is_relative_to(Path(sys.prefix).resolve())" in payload.read_text()
    assert "native_cpu_adjoint_available()" in payload.read_text()
    assert (tmp_path / "main-smoke-tests/test_smoke_baseline.py").read_bytes() == (
        ROOT / "tests/smoke/test_smoke_baseline.py"
    ).read_bytes()
