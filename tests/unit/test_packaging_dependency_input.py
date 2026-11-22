"""Complete optional packaging input and real local package-tool consumers."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from importlib import import_module
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = REFERENCE / "packaging-command-deps"


def test_complete_packaging_input_and_active_installed_closure() -> None:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    raw = (REFERENCE / "dev_tools/requirements/deps/packaging.txt").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "b6ad665bf1ecec33e4af47c5097e25833011e0c5"
    )
    requirements = [
        Requirement(line)
        for line in raw.decode().splitlines()
        if line and not line.startswith("#")
    ]
    assert [str(requirement) for requirement in requirements] == [
        "virtualenv",
        "grpcio-tools",
        "setuptools",
        "wheel",
        "twine",
    ]
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions(path=[str(DEPENDENCIES)])
    }
    lock = (REFERENCE / "dev_tools/requirements/envs/dev.env.txt").read_text()
    for requirement in requirements:
        assert (
            installed[canonicalize_name(requirement.name)].version
            in requirement.specifier
        )
    assert len(installed) == 32
    for name, distribution in installed.items():
        notices = [
            path
            for path in distribution.files or []
            if any(term in str(path).lower() for term in ("license", "copying"))
        ]
        if name == "grpcio-tools":
            assert notices == []
            assert distribution.metadata["License"] == "Apache License 2.0"
        else:
            assert notices, name
        if name == "setuptools":
            assert distribution.version == "80.9.0"
        else:
            assert f"{name}=={distribution.version}\n" in lock
        for line in distribution.requires or []:
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            assert (
                installed[canonicalize_name(requirement.name)].version
                in requirement.specifier
            )


def test_actual_isolated_virtualenv_interpreter(tmp_path: Path) -> None:
    destination = tmp_path / "isolated python"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "virtualenv",
            "--no-seed",
            "--no-download",
            "--no-periodic-update",
            str(destination),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    executable = destination / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    probe = subprocess.run(
        [
            str(executable),
            "-I",
            "-c",
            "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0, probe.stderr
    prefix, base = json.loads(probe.stdout)
    assert Path(prefix).resolve() == destination.resolve()
    assert Path(base).resolve() != destination.resolve()


def test_actual_wheel_container_and_offline_validation(tmp_path: Path) -> None:
    unpacked = tmp_path / "package"
    metadata = unpacked / "dependency_probe-1.0.dist-info"
    metadata.mkdir(parents=True)
    (unpacked / "dependency_probe.py").write_text("VALUE = 1\n")
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dependency-probe\nVersion: 1.0\n"
        "Summary: Local packaging-tool fixture\n"
        "Description-Content-Type: text/plain\n\nA local packaging-tool fixture.\n"
    )
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: dependency-input-test\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    packed = subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert packed.returncode == 0, packed.stdout + packed.stderr
    wheels = tuple(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert archive.read("dependency_probe.py") == b"VALUE = 1\n"
        assert "dependency_probe-1.0.dist-info/RECORD" in archive.namelist()
    checked = subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", str(wheels[0])],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "PASSED" in checked.stdout


def test_real_native_sdist_passes_offline_package_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    name = import_module("setuptools.build_meta").build_sdist(
        str(tmp_path),
        config_settings={"--global-option": ["egg_info", "--egg-base", str(tmp_path)]},
    )
    assert isinstance(name, str)
    path = tmp_path / name
    with tarfile.open(path, "r:gz") as archive:
        prefix = name.removesuffix(".tar.gz") + "/"
        for name in ("pyproject.toml", "README.md", "MANIFEST.in"):
            stream = archive.extractfile(prefix + name)
            assert stream is not None
            assert stream.read() == (ROOT / name).read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", str(path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout
