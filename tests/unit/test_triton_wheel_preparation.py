"""Exercise bounded wheel preparation with real pip and local synthetic wheels."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _wheel(directory: Path, name: str, dependency: str = "") -> None:
    version = "3.9.0" if name == "triton" else "1.0"
    info = f"{name}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    with zipfile.ZipFile(
        directory / f"{name}-{version}-py3-none-any.whl", "w"
    ) as wheel:
        wheel.writestr(f"{info}/METADATA", metadata)
        wheel.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr(f"{info}/RECORD", "")


@pytest.fixture(scope="module")
def pip_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("triton-pip") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(directory)], check=True, timeout=45
    )
    return directory / "bin/python"


def _prepare(
    directory: Path,
    pip_python: Path,
    *,
    fail_first: bool = False,
    index: str = "https://mirror.invalid/simple",
    leaf: bool = True,
    restored: bool = False,
) -> subprocess.CompletedProcess[str]:
    public = directory / "public"
    public.mkdir()
    seeded = directory / "managed" if restored else public
    seeded.mkdir(exist_ok=True)
    _wheel(seeded, "triton", "cache_leaf>=1")
    if leaf:
        _wheel(seeded, "cache_leaf")
    (directory / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncuda = ["triton>=3.1,<4"]\n'
    )
    bindir = directory / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text(f'#!/bin/sh\nexec "{pip_python}" "$@"\n')
    python.chmod(0o755)
    timeout = bindir / "timeout"
    timeout.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$FQ_TIMEOUT_LOG"\n'
        'if [ "$FQ_FAIL_FIRST" = 1 ] && [ ! -f "$FQ_TIMEOUT_LOG.failed" ]; then\n'
        '  touch "$FQ_TIMEOUT_LOG.failed"\n  exit 124\nfi\n'
        'shift 2\nexec "$@"\n'
    )
    timeout.chmod(0o755)
    action = yaml.safe_load(
        (ROOT / ".github/actions/prepare-triton-wheels/action.yml").read_text()
    )
    return subprocess.run(
        ["bash", "-c", action["runs"]["steps"][0]["run"]],
        cwd=directory,
        env={
            **os.environ,
            "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
            "RUNNER_ENVIRONMENT": "github-hosted" if restored else "self-hosted",
            "FQ_TRITON_WHEEL_DIRECTORY": str(directory / "managed"),
            "FQ_TRITON_PUBLIC_WHEELS": str(public),
            "FQ_FAIL_FIRST": "1" if fail_first else "0",
            "FQ_TIMEOUT_LOG": str(directory / "attempts.log"),
            "PIP_NO_INDEX": "1",
            "PIP_INDEX_URL": index,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(directory / "pip-cache"),
        },
        text=True,
        capture_output=True,
        timeout=45,
    )


@pytest.mark.parametrize("restored", [False, True])
def test_cached_wheels_preserve_the_complete_dependency_closure(
    tmp_path: Path, pip_python: Path, restored: bool
) -> None:
    result = _prepare(tmp_path, pip_python, restored=restored)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in (tmp_path / "managed").glob("*.whl")} == {
        "triton-3.9.0-py3-none-any.whl",
        "cache_leaf-1.0-py3-none-any.whl",
    }
    assert len((tmp_path / "attempts.log").read_text().splitlines()) == 1


def test_a_slow_mirror_falls_back_once_with_a_separate_budget(
    tmp_path: Path, pip_python: Path
) -> None:
    result = _prepare(tmp_path, pip_python, fail_first=True)
    assert result.returncode == 0, result.stdout + result.stderr
    mirror, official = (tmp_path / "attempts.log").read_text().splitlines()
    assert "180s" in mirror and "https://mirror.invalid/simple" in mirror
    assert "300s" in official and "https://pypi.org/simple" in official


def test_a_failed_official_index_is_not_retried_as_a_fallback(
    tmp_path: Path, pip_python: Path
) -> None:
    result = _prepare(
        tmp_path, pip_python, fail_first=True, index="https://pypi.org/simple"
    )
    assert result.returncode != 0
    assert len((tmp_path / "attempts.log").read_text().splitlines()) == 1


def test_an_incomplete_cache_cannot_pass_preparation(
    tmp_path: Path, pip_python: Path
) -> None:
    result = _prepare(tmp_path, pip_python, leaf=False)
    assert result.returncode != 0
    assert "No matching distribution" in result.stderr


def test_workflow_prepares_wheels_before_installing_the_project() -> None:
    job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"][
        "triton-optional"
    ]
    assert job["timeout-minutes"] == 15
    steps = job["steps"]
    preparation = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/prepare-triton-wheels"
    )
    install = next(step for step in steps if "-e '.[dev,cuda]'" in step.get("run", ""))
    assert steps.index(preparation) < steps.index(install)
    assert preparation["timeout-minutes"] == 9
    assert install["timeout-minutes"] == 4
    assert any(step.get("uses") == "actions/cache@v4" for step in steps)
    assert any("-m pip check" in step.get("run", "") for step in steps)
