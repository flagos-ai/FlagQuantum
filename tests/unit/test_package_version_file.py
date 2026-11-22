"""Frozen version-file responsibility and native release metadata consistency."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.version import VERSION_INFO, __version__, get_version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.ftqc_reference
def test_complete_source_version_file() -> None:
    """The original module consists solely of its independent version literal."""
    source = import_module("qualtran._version")
    assert source.__file__ is not None
    raw = Path(source.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "a21f8f56cc9e68b4e36f36665fdeba877ced3630"
    )
    (assignment,) = ast.parse(raw).body
    assert isinstance(assignment, ast.Assign)
    (target,) = assignment.targets
    assert isinstance(target, ast.Name) and target.id == "__version__"
    assert ast.literal_eval(assignment.value) == source.__version__ == "0.7.0"


def test_native_release_version_has_one_metadata_owner() -> None:
    """Public import, callable and build configuration resolve the native owner."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert configuration["project"]["dynamic"] == ["version"]
    assert configuration["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "flagquantum.version.__version__"
    }
    assert fq.__version__ == get_version() == __version__
    assert VERSION_INFO["release_level"] == "final"
    assert tuple(map(int, __version__.split("."))) == tuple(
        VERSION_INFO[part] for part in ("major", "minor", "patch")
    )
