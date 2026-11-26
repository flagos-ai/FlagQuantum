"""Complete frozen lock compiler order, failure semantics and real offline locks."""

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import process
from tools.development import requirement_locks as native
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = REFERENCE / "dev_tools/requirements/re-pip-compile.sh"
DEPENDENCIES = REFERENCE / "requirement-lock-deps"


def environment() -> dict[str, str]:
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(DEPENDENCIES), str(ROOT)]),
        PATH=os.pathsep.join([str(DEPENDENCIES / "bin"), os.environ["PATH"]]),
        PIP_NO_INDEX="1",
        PIP_FIND_LINKS=str(REFERENCE / "requirement-lock-wheels"),
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_TOOLS_CACHE_DIR=str(REFERENCE / "requirement-lock-cache"),
    )


def profiles(root: Path) -> None:
    (root / "deps").mkdir(parents=True)
    for _, names in native.PROFILES:
        for name in names:
            (root / "deps" / f"{name}.txt").write_text("packaging==25.0\n")


def outputs(root: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(root.glob("*.env.txt"))}


def test_complete_source_and_exact_optional_lock() -> None:
    checked = set()
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in {"Q-0099", "Q-0107", "Q-0112"}:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                checked.add(row["id"])
    assert checked == {"Q-0099", "Q-0107", "Q-0112"}
    assert (
        REFERENCE / "dev_tools/requirements/deps/pip-tools.txt"
    ).read_text().strip() == "pip-tools"
    installed = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(DEPENDENCIES)])
    }
    assert installed == {
        "build": "1.3.0",
        "click": "8.3.1",
        "packaging": "25.0",
        "pip-tools": "7.5.2",
        "pyproject-hooks": "1.2.0",
        "tomli": "2.3.0",
        "wheel": "0.45.1",
        "pip": "25.3",
        "setuptools": "80.9.0",
    }
    lock = (REFERENCE / "dev_tools/requirements/envs/pip-tools.env.txt").read_text()
    for name, version in installed.items():
        if name not in {"pip", "setuptools"}:
            assert f"{name}=={version}\n" in lock


def test_complete_original_and_native_command_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles(tmp_path)
    binary = tmp_path / "bin"
    binary.mkdir()
    compiler = binary / "pip-compile"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\nfrom pathlib import Path\n"
        "with Path('calls.jsonl').open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "output=sys.argv[1].split('=',1)[1]\nPath(output).write_text('packaging==25.0\\n')\n"
    )
    compiler.chmod(0o755)
    subprocess.run(
        ["/bin/bash", str(SOURCE)],
        cwd=tmp_path,
        env=dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"]),
        check=True,
        timeout=30,
        capture_output=True,
    )
    source_calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    native_calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        native_calls.append(arguments[1:])
        assert kwargs == dict(cwd=kwargs["cwd"], check=False, timeout=1200)
        target = Path(kwargs["cwd"]) / arguments[1].split("=", 1)[1]
        target.write_text("packaging==25.0\n")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(process, "run", run)
    assert native.compile_locks(tmp_path, tmp_path / "native") == 0
    assert source_calls == native_calls
    assert len(source_calls) == 8
    assert outputs(tmp_path / "envs") == outputs(tmp_path / "native")


def test_real_offline_source_native_locks_and_cli(tmp_path: Path) -> None:
    profiles(tmp_path)
    (tmp_path / "deps/runtime.txt").write_text("-r shared.txt\n")
    (tmp_path / "deps/shared.txt").write_text("packaging==25.0\n")
    subprocess.run(
        ["/bin/bash", str(SOURCE)],
        cwd=tmp_path,
        env=environment(),
        check=True,
        capture_output=True,
        timeout=180,
    )
    target = tmp_path / "native output"
    command = [
        sys.executable,
        "-m",
        "tools.development.requirement_locks",
        "--source-root",
        str(tmp_path),
        "--output",
        str(target),
        "--pip-compile",
        str(DEPENDENCIES / "bin/pip-compile"),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs(target) == outputs(tmp_path / "envs")
    assert len(outputs(target)) == 8
    assert all("packaging==25.0" in text for text in outputs(target).values())
    before = {path.name: path.stat().st_mtime_ns for path in target.iterdir()}
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name: path.stat().st_mtime_ns for path in target.iterdir()} == before


@pytest.mark.parametrize("failed", range(8))
def test_each_compiler_failure_keeps_target_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: int
) -> None:
    profiles(tmp_path)
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "dev.env.txt").write_text("old\n")
    calls = []
    staged_paths = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stage = Path(kwargs["cwd"])
        staged_paths.append(stage)
        calls.append(arguments)
        if len(calls) - 1 == failed:
            return subprocess.CompletedProcess(arguments, 9)
        (stage / arguments[1].split("=", 1)[1]).write_text("new\n")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(process, "run", run)
    assert native.compile_locks(tmp_path, destination) == 9
    assert len(calls) == failed + 1
    assert outputs(destination) == {"dev.env.txt": "old\n"}
    assert all(not path.exists() for path in staged_paths)


def test_missing_timeout_and_output_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="Missing requirement profile"):
        native.compile_locks(tmp_path, tmp_path / "out")
    profiles(tmp_path)
    stages = []

    def timeout(arguments: list[str], **kwargs: Any) -> Any:
        stages.append(Path(kwargs["cwd"]))
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    monkeypatch.setattr(process, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        native.compile_locks(tmp_path, tmp_path / "out")
    assert all(not path.exists() for path in stages)

    def missing(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(process, "run", missing)
    with pytest.raises(FileNotFoundError):
        native.compile_locks(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_destination_pins_seed_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    for profile, _ in native.PROFILES:
        (output / f"{profile}.env.txt").write_text("packaging==25.0\n")
    (output / "unrelated.txt").write_text("preserved\n")

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        target = Path(kwargs["cwd"]) / arguments[1].split("=", 1)[1]
        assert target.read_text() == "packaging==25.0\n"
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(process, "run", run)
    assert native.compile_locks(tmp_path, output) == 0
    assert (output / "unrelated.txt").read_text() == "preserved\n"


@pytest.mark.parametrize("failed", range(8))
def test_original_failure_stops_at_each_command_with_partial_outputs(
    tmp_path: Path, failed: int
) -> None:
    profiles(tmp_path)
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "pip-compile"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "counter=Path('counter.txt')\n"
        "index=int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(index+1))\n"
        f"if index == {failed}: sys.exit(9)\n"
        "Path(sys.argv[1].split('=',1)[1]).write_text('complete\\n')\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(SOURCE)],
        cwd=tmp_path,
        env=dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 9
    assert (tmp_path / "counter.txt").read_text() == str(failed + 1)
    assert len(outputs(tmp_path / "envs")) == failed


def test_all_optional_tool_dependencies_resolve_with_frozen_pins() -> None:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions(path=[str(DEPENDENCIES)])
    }
    # Check both the original lock's Python marker environment and this runner.
    for version, patch in (("3.10", "3.10.19"), ("3.12", "3.12.14")):
        marker_environment = {
            key: str(value) for key, value in default_environment().items()
        }
        marker_environment.update(
            python_version=version, python_full_version=patch, extra=""
        )
        for distribution in installed.values():
            for declaration in distribution.requires or ():
                requirement = Requirement(declaration)
                if requirement.marker is not None and not requirement.marker.evaluate(
                    marker_environment
                ):
                    continue
                dependency = installed[canonicalize_name(requirement.name)]
                assert dependency.version in requirement.specifier


def test_original_lock_patch_level_dependency_boundary() -> None:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    distributions = list(importlib.metadata.distributions(path=[str(DEPENDENCIES)]))
    build = next(
        distribution
        for distribution in distributions
        if distribution.metadata["Name"] == "build"
    )
    requirement = next(
        Requirement(value)
        for value in build.requires or ()
        if value.startswith("importlib-metadata")
    )
    environment = {key: str(value) for key, value in default_environment().items()}
    environment.update(python_version="3.10", python_full_version="3.10.0", extra="")
    assert requirement.marker is not None and requirement.marker.evaluate(environment)
    assert (
        "importlib-metadata=="
        not in (REFERENCE / "dev_tools/requirements/envs/pip-tools.env.txt").read_text()
    )
