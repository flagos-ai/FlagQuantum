"""Exercise complete CUDA-Q cache resolution with real pip and tiny local wheels."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ("0.15.1", "0.16.0.post1")


def _wheel(
    directory: Path, name: str, version: str, dependencies: tuple[str, ...]
) -> None:
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {item}\n" for item in dependencies)
    with zipfile.ZipFile(
        directory / f"{normalized}-{version}-py3-none-any.whl", "w"
    ) as archive:
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{info}/RECORD", "")


@pytest.fixture(scope="module")
def pip_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    environment = tmp_path_factory.mktemp("cudaq-pip") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)], check=True, timeout=45
    )
    return environment / "bin/python"


def _prepare(
    tmp_path: Path, pip_python: Path, public: Path, managed: Path, host: str
) -> subprocess.CompletedProcess[str]:
    action = yaml.safe_load(
        (ROOT / ".github/actions/prepare-cudaq-wheels/action.yml").read_text()
    )
    script = action["runs"]["steps"][0]["run"]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text(f'#!/bin/sh\nexec "{pip_python}" "$@"\n')
    python.chmod(0o755)
    # Mirror probes remain offline in this test; never contact a public index.
    timeout = bindir / "timeout"
    timeout.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    timeout.chmod(0o755)
    return subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
            "OPTIONAL_VERSIONS": " ".join(VERSIONS),
            "FQ_CUDAQ_WHEEL_DIRECTORY": str(managed),
            "FQ_CUDAQ_PUBLIC_WHEELS": str(public),
            "FQ_RUNNER_KIND": "",
            "RUNNER_ENVIRONMENT": host,
            "PIP_NO_INDEX": "1",
            "PIP_INDEX_URL": "https://mirror.invalid/simple",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
        },
        text=True,
        capture_output=True,
        timeout=60,
    )


def _populate(directory: Path, *, leaf: bool = True, flavor: str = "cu13") -> None:
    directory.mkdir()
    for version in VERSIONS:
        _wheel(directory, "cudaq", version, (f"cuda-quantum-{flavor}=={version}",))
        _wheel(directory, f"cuda-quantum-{flavor}", version, ("cache-bridge==2",))
    _wheel(directory, "cache-bridge", "2", ("cache-leaf==1",))
    if leaf:
        _wheel(directory, "cache-leaf", "1", ())


def test_self_hosted_cache_works_without_runner_kind_and_contains_transitive_leaves(
    tmp_path: Path,
    pip_python: Path,
) -> None:
    public, managed = tmp_path / "public", tmp_path / "managed"
    _populate(public)
    result = _prepare(tmp_path, pip_python, public, managed, "self-hosted")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Using the available self-hosted public wheel cache" in result.stdout
    assert "trying official PyPI" not in result.stdout
    assert {p.name for p in managed.glob("*.whl")} == {
        p.name for p in public.glob("*.whl")
    }


def test_restored_managed_cache_is_offline_on_hosted_runners(
    tmp_path: Path,
    pip_python: Path,
) -> None:
    managed = tmp_path / "managed"
    _populate(managed)
    result = _prepare(
        tmp_path, pip_python, tmp_path / "absent", managed, "github-hosted"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Complete offline CUDA-Q") == 2
    assert "trying official PyPI" not in result.stdout


@pytest.mark.parametrize("failure", ["missing_transitive_leaf", "wrong_cuda_flavor"])
def test_incomplete_or_incompatible_cache_is_not_a_success(
    tmp_path: Path,
    pip_python: Path,
    failure: str,
) -> None:
    public = tmp_path / "public"
    _populate(
        public,
        leaf=failure != "missing_transitive_leaf",
        flavor="cu12" if failure == "wrong_cuda_flavor" else "cu13",
    )
    result = _prepare(tmp_path, pip_python, public, tmp_path / "managed", "self-hosted")
    assert result.returncode != 0
    assert "Complete offline CUDA-Q" not in result.stdout
    assert "trying official PyPI" in result.stdout
    assert (
        "No matching distribution" in result.stderr
        or "ResolutionImpossible" in result.stderr
    )


def test_certified_checks_and_aggregate_are_retained() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    job = jobs["cudaq-optional"]
    assert job["env"]["OPTIONAL_VERSIONS"].split() == list(VERSIONS)
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    steps = job["steps"]
    preparation = next(
        s for s in steps if s.get("uses") == "./.github/actions/prepare-cudaq-wheels"
    )
    assert preparation["timeout-minutes"] == 10
    assert any(s.get("uses") == "./.github/actions/prepare-cudaq-wheels" for s in steps)
    check = next(
        s["run"]
        for s in steps
        if s.get("name") == "Verify both certified CUDA-Q versions"
    )
    for command in (
        "for version in ${OPTIONAL_VERSIONS}; do",
        "python -m pip check",
        "python tools/check_cudaq_export_contract.py --verify-sdk",
        "python -m pytest tests/test_cudaq_export.py tests/test_cudaq_export_conformance.py -m cudaq -q",
        "assert not any(name == 'cudaq' or name.startswith('cudaq.') for name in sys.modules)",
    ):
        assert command in check
    assert "--no-deps" not in check
    assert "continue-on-error" not in job
    assert "cudaq-optional" in jobs["ci-complete"]["needs"]


@pytest.mark.parametrize(
    "index,mirror_result,official_result,durations,expected_code",
    [
        ("https://pypi.org/simple", 0, 0, ["300s"], 0),
        ("https://pypi.org/simple/", 0, 1, ["300s"], 1),
        ("https://mirror.invalid/simple", 0, 0, ["180s"], 0),
        ("https://mirror.invalid/simple", 1, 0, ["180s", "300s"], 0),
        ("https://mirror.invalid/simple", 124, 1, ["180s", "300s"], 1),
    ],
)
def test_download_deadlines_and_official_fallback_do_not_repeat_an_endpoint(
    tmp_path: Path,
    index: str,
    mirror_result: int,
    official_result: int,
    durations: list[str],
    expected_code: int,
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/prepare-cudaq-wheels/action.yml").read_text()
    )
    prefix = action["runs"]["steps"][0]["run"].split(
        "for version in ${OPTIONAL_VERSIONS};", 1
    )[0]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    timeout = bindir / "timeout"
    timeout.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TIMEOUT_CALLS'], 'a') as log: log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "raise SystemExit(int(os.environ['MIRROR_RESULT'] if sys.argv[2] == '180s' else os.environ['OFFICIAL_RESULT']))\n"
    )
    timeout.chmod(0o755)
    completed = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            prefix + "\ndownload_wheels --only-binary=:all: cuda-quantum-cu13==0.15.1",
        ],
        env={
            **os.environ,
            "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
            "PIP_INDEX_URL": index,
            "FQ_CUDAQ_WHEEL_DIRECTORY": str(tmp_path / "wheels"),
            "OPTIONAL_VERSIONS": VERSIONS[0],
            "RUNNER_ENVIRONMENT": "github-hosted",
            "TIMEOUT_CALLS": str(calls),
            "MIRROR_RESULT": str(mirror_result),
            "OFFICIAL_RESULT": str(official_result),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == expected_code, completed.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [args[1] for args in commands] == durations
    assert all(args[0] == "--kill-after=15s" for args in commands)
    indexes = [args[args.index("--index-url") + 1] for args in commands]
    assert indexes == (
        [index]
        if durations == ["180s"]
        else [index, "https://pypi.org/simple"]
        if len(durations) == 2
        else ["https://pypi.org/simple"]
    )
