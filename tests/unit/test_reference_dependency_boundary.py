"""Missing SDKs cannot silently remove mandatory reference comparisons."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._reference_dependencies import unavailable_reference

pytestmark = pytest.mark.unit


def test_optional_absence_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES", raising=False)
    error = ModuleNotFoundError(
        "The reference SDK is absent", name="missing_reference_sdk"
    )
    with pytest.raises(pytest.skip.Exception, match="Optional reference dependency"):
        unavailable_reference(error)


@pytest.mark.parametrize("required", ["1", "unexpected"])
def test_required_absence_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch, required: str
) -> None:
    monkeypatch.setenv("FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES", required)
    cause = RuntimeError("Original dependency cause")
    error = ModuleNotFoundError(
        "The reference SDK is absent", name="missing_reference_sdk"
    )
    error.__cause__ = cause
    with pytest.raises(ModuleNotFoundError) as failure:
        unavailable_reference(error)
    assert failure.value is error and failure.value.__cause__ is cause
    assert failure.value.name == "missing_reference_sdk"


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("present", [False, True])
def test_real_collection_and_execution_obey_dependency_mode(
    tmp_path: Path, required: bool, present: bool
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_control.py").write_text(
        "def test_control():\n    assert 2 + 3 == 5\n"
    )
    dependency = "fq_test_reference_boundary_dependency"
    if present:
        (tmp_path / (dependency + ".py")).write_text("VALUE = 11\n")
    (tmp_path / "test_comparison.py").write_text(
        "from tests._reference_dependencies import unavailable_reference\n"
        "try:\n"
        f"    import {dependency} as sdk\n"
        "except ModuleNotFoundError as error:\n"
        "    unavailable_reference(error)\n"
        "def test_original_comparison():\n"
        "    assert sdk.VALUE == 11\n"
    )
    environment = dict(
        os.environ,
        FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES="1" if required else "0",
        PYTHONPATH=os.pathsep.join([str(tmp_path), *sys.path]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-c",
            str(tmp_path / "pytest.ini"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = result.stdout + result.stderr
    if present:
        assert result.returncode == 0 and "2 passed" in output, output
        assert "skipped" not in output
    elif required:
        assert result.returncode == 2, output
        assert "ModuleNotFoundError" in output and dependency in output
        assert "skipped" not in output
    else:
        assert result.returncode == 0 and "1 passed, 1 skipped" in output, output
