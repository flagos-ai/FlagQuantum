"""The post-merge smoke tests the installed wheel outside its source checkout."""

from __future__ import annotations

import os
import shutil
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
        if step.get("name") == "Install and smoke-test the wheel in a clean runtime"
    )
    checkout = tmp_path / "checkout"
    sources = (
        "tests/smoke/test_smoke_baseline.py",
        "tools/check_main_smoke_qec_wheel.py",
        "examples/qec/ai_predecoder.py",
    )
    for name in sources:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    wheel = "dist/flagquantum-0.4.0-cp310-abi3-linux_x86_64.whl"
    (checkout / wheel).parent.mkdir()
    (checkout / wheel).touch()

    tools = tmp_path / "bin"
    tools.mkdir()
    python = tools / "python"
    python.write_text(
        "#!/bin/sh\n"
        'printf "%s|%s|%s\\n" "$PWD" "$0" "$*" >> "$CALLS"\n'
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        '  mkdir -p "$3/bin" && cp "$0" "$3/bin/python"\n'
        'elif [ "$1" = "-I" ] && [ "$2" = "-" ]; then\n'
        '  cat > "$PAYLOAD"\n'
        "fi\n"
    )
    python.chmod(0o755)
    calls = tmp_path / "calls"
    payload = tmp_path / "payload"
    completed = subprocess.run(
        ["bash", "-eu", "-c", command],
        cwd=checkout,
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
    (smoke_dir,) = tmp_path.glob("main-smoke-tests.*")
    smoke_python = (smoke_dir / "venv/bin/python").resolve()
    recorded = [
        (Path(cwd).resolve(), Path(executable).resolve(), args)
        for cwd, executable, args in (
            line.split("|", 2) for line in calls.read_text().splitlines()
        )
    ]
    assert recorded[0] == (
        checkout.resolve(),
        python.resolve(),
        f"-m venv {smoke_dir / 'venv'}",
    )
    install = recorded[1:5]
    assert all(
        cwd == checkout.resolve() and exe == smoke_python for cwd, exe, _ in install
    )
    assert [args for _, _, args in install] == [
        "-m pip install torch>=2.10,<2.11 --index-url https://download.pytorch.org/whl/cpu",
        "-m pip install pytest",
        f"-m pip install --no-deps {wheel}",
        "-m pip check",
    ]
    assert recorded[5:] == [
        (smoke_dir.resolve(), smoke_python, "-I -"),
        (smoke_dir.resolve(), smoke_python, "-I -m pytest test_smoke_baseline.py -q"),
        (smoke_dir.resolve(), smoke_python, "-I check_main_smoke_qec_wheel.py"),
    ]
    probe = payload.read_text()
    assert "is_relative_to(Path(sys.prefix).resolve())" in probe
    assert "native_cpu_adjoint_available()" in probe
    for name in sources:
        assert (smoke_dir / Path(name).name).read_bytes() == (ROOT / name).read_bytes()
