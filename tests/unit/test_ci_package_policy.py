import ast
import importlib.util
import re
from pathlib import Path

import pytest

from tools.artifact_manifest import main as manifest_main
from tools.verify_distribution_artifacts import (
    REQUIRED_MEMBER_SUFFIXES,
    _forbidden,
    _missing_required_members,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_package_uses_one_dynamic_version_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "flagquantum.version.__version__"}' in pyproject
    assert '\nversion = "0.1.0"' not in pyproject


def test_ci_python_paths_resolve():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    paths = set(re.findall(r"\bflagquantum/[A-Za-z0-9_./-]+", workflow))
    assert paths
    assert not [path for path in paths if not (ROOT / path).exists()]


def test_ci_install_checks_import_existing_modules():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    snippets = re.findall(r'python -c "([^"\n]+)"', workflow)
    modules = {
        node.module
        for snippet in snippets
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("flagquantum.")
    }
    assert modules
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module


def test_distribution_quarantine_rejects_repo_only_members():
    assert _forbidden("flagquantum/__pycache__/module.pyc")
    assert _forbidden("source/tests/test_api.py")
    assert _forbidden("source/examples/data.parquet")
    assert not _forbidden("flagquantum/core/ir.py")


def test_distribution_requires_runtime_profiles_and_numerical_contract() -> None:
    assert _missing_required_members(REQUIRED_MEMBER_SUFFIXES) == ()
    assert _missing_required_members(REQUIRED_MEMBER_SUFFIXES[1:]) == (
        "flagquantum/simulation/numerics/double-single-contract.toml",
    )


def test_artifact_manifest_records_checksum(tmp_path):
    artifact = tmp_path / "sample.whl"
    artifact.write_bytes(b"flagquantum")
    output = tmp_path / "manifest.json"

    assert manifest_main(["--output", str(output), str(artifact)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "flagquantum_artifact_manifest_v1" in text
    assert "sample.whl" in text
    assert "sha256" in text
