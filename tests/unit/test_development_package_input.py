"""Complete developer build declaration and actual source-wheel boundary."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def test_complete_development_build_declaration() -> None:
    raw = (REFERENCE / "dev_tools/pyproject.toml").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "46f067567919a8176e943f1f9e9fa2d08e17d427"
    )
    assert tomllib.loads(raw.decode()) == {
        "build-system": {
            "requires": ["setuptools>=61.0"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": "qualtran-dev-tools",
            "version": "0.1.0",
            "description": "Local development tools for the Qualtran project.",
            "requires-python": ">=3.10",
        },
        "tool": {"setuptools": {"packages": ["qualtran_dev_tools"]}},
    }


def test_actual_development_wheel_and_retained_nested_omission(tmp_path: Path) -> None:
    source = REFERENCE / "dev_tools"
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(source / "pyproject.toml", project / "pyproject.toml")
    shutil.copytree(
        source / "qualtran_dev_tools",
        project / "qualtran_dev_tools",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    output = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; "
            "import sys; build_wheel(sys.argv[1])",
            str(output),
        ],
        cwd=project,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        files = archive.namelist()
        metadata = Parser().parsestr(
            archive.read("qualtran_dev_tools-0.1.0.dist-info/METADATA").decode()
        )
        assert metadata["Name"] == "qualtran-dev-tools"
        assert metadata["Version"] == "0.1.0"
        assert metadata["Requires-Python"] == ">=3.10"
        assert metadata.get_all("Requires-Dist", []) == []
        for path in (source / "qualtran_dev_tools").glob("*.py"):
            relative = path.relative_to(source).as_posix()
            assert archive.read(relative) == path.read_bytes()
        assert (source / "qualtran_dev_tools/make_reference_docs/__init__.py").is_file()
        assert not any("/make_reference_docs/" in name for name in files)
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.util
import json
import sys
sys.path.insert(0, sys.argv[1])
from qualtran_dev_tools.shell_tools import output_of
print(json.dumps([
    output_of([sys.executable, "-c", "print(42)"], timeout=5),
    importlib.util.find_spec("qualtran_dev_tools.make_reference_docs") is None,
]))
""",
            str(wheels[0]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert json.loads(probe.stdout) == ["42", True]


def test_native_checkout_developer_library_without_separate_distribution(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import json
import sys
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[3])
sys.path.insert(0, sys.argv[1])
from tools.development.files import WriteIfDifferent
from tools.development.process import output_of
from tools.development import reference_templates

target = Path(sys.argv[2])
with WriteIfDifferent(target) as stream:
    stream.write("42\\n")
print(json.dumps([
    target.read_text(),
    output_of([sys.executable, "-c", "print(42)"], timeout=5),
    reference_templates.__name__,
]))
""",
            str(ROOT),
            str(tmp_path / "report.txt"),
            json.dumps(sys.path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    notice, payload = result.stdout.splitlines()
    assert notice == f"{tmp_path / 'report.txt'}\t writing."
    assert json.loads(payload) == [
        "42\n",
        "42",
        "tools.development.reference_templates",
    ]
