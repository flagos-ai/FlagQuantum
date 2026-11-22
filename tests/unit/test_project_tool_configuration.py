"""Complete frozen tool configuration and native pytest policy consumers."""

import os
import subprocess
import sys
from hashlib import sha1
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"


@pytest.mark.ftqc_reference
def test_complete_source_configuration_tables() -> None:
    raw = (REFERENCE / "pyproject.toml").read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "7b1548955775f38d5f36cae38cc0346dbf9b46d8"
    )
    configuration = tomllib.loads(raw.decode())
    assert set(configuration) == {"tool"}
    assert configuration["tool"] == {
        "black": {
            "line-length": 100,
            "target_version": ["py39", "py310", "py311", "py312"],
            "skip-string-normalization": True,
            "skip-magic-trailing-comma": True,
            "exclude": "qualtran/protos/*",
        },
        "isort": {
            "profile": "black",
            "order_by_type": False,
            "line_length": 100,
            "remove_redundant_aliases": True,
            "skip_glob": ["qualtran/protos/*"],
        },
        "pytest": {
            "ini_options": {
                "filterwarnings": [
                    "ignore::DeprecationWarning:quimb.linalg.approx_spectral:",
                    "ignore:.*standard platformdirs.*:DeprecationWarning:jupyter_client.*",
                ],
                "python_classes": "*TestSuite",
                "addopts": "--strict-markers",
                "markers": [
                    "slow: slow tests.",
                    "notebook: unit tests that execute a notebook.",
                ],
            }
        },
    }


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("profile", ("source", "native"))
@pytest.mark.parametrize("unknown_marker", (False, True))
def test_actual_pytest_configuration_consumption(
    tmp_path: Path, profile: str, unknown_marker: bool
) -> None:
    source = (
        REFERENCE / "pyproject.toml" if profile == "source" else ROOT / "pytest.ini"
    )
    (tmp_path / source.name).write_bytes(source.read_bytes())
    fixture = tmp_path / "test_probe.py"
    fixture.write_text(
        "import pytest\n"
        "import warnings\n\n"
        "class ProbeTestSuite:\n"
        "    def test_source_class(self):\n"
        "        pass\n\n"
        "class TestProbe:\n"
        "    def test_native_class(self):\n"
        "        pass\n\n"
        "def test_warning_policy():\n"
        "    warnings.warn_explicit('spectral notice', DeprecationWarning,\n"
        "                           'spectral.py', 1, module='quimb.linalg.approx_spectral')\n"
        "    warnings.warn_explicit('standard platformdirs notice', DeprecationWarning,\n"
        "                           'client.py', 1, module='jupyter_client.paths')\n"
        "    warnings.warn_explicit('unrelated notice', DeprecationWarning,\n"
        "                           'other.py', 1, module='other')\n"
        + (
            "\n@pytest.mark.undeclared_marker\ndef test_unknown_marker():\n    pass\n"
            if unknown_marker
            else ""
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(fixture)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if unknown_marker:
        assert result.returncode == 2, output
        assert "undeclared_marker" in output and "not found in" in output
        return
    assert result.returncode == 0, output
    assert "2 passed" in output
    assert "unrelated notice" in output
    if profile == "source":
        assert "ProbeTestSuite::test_source_class PASSED" in output
        assert "TestProbe::test_native_class" not in output
        assert "spectral notice" not in output
        assert "platformdirs notice" not in output
    else:
        assert "TestProbe::test_native_class PASSED" in output
        assert "ProbeTestSuite::test_source_class" not in output
        assert "spectral notice" in output
        assert "platformdirs notice" in output
