from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_dependency_policy import load_toml, policy_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    return (
        load_toml(ROOT / "dependency-policy.toml"),
        load_toml(ROOT / "pyproject.toml"),
    )


def test_dependency_policy_matches_packaging_exactly() -> None:
    assert policy_errors(*_inputs()) == ()


def test_dependency_policy_rejects_unclassified_or_drifting_extras() -> None:
    policy, pyproject = _inputs()
    pyproject["project"]["optional-dependencies"]["new-provider"] = [
        "new-provider-sdk>=1"
    ]
    errors = policy_errors(policy, pyproject)
    assert any("policy extras differ from pyproject" in error for error in errors)
    assert any("extra classification is incomplete" in error for error in errors)


def test_dependency_policy_rejects_core_framework_contamination() -> None:
    policy, pyproject = _inputs()
    pyproject["project"]["dependencies"].append("qiskit>=2")
    errors = policy_errors(policy, pyproject)
    assert any("core dependencies must contain only torch" in error for error in errors)


def test_dependency_policy_reports_malformed_requirements() -> None:
    policy, pyproject = _inputs()
    pyproject["project"]["optional-dependencies"]["qiskit"] = [""]
    errors = policy_errors(policy, pyproject)
    assert "extra 'qiskit' must be a list of valid requirements" in errors


def test_dependency_policy_rejects_aggregate_drift() -> None:
    policy, pyproject = _inputs()
    policy["extras"]["interop-all"] = copy.deepcopy(policy["extras"]["interop-all"])[
        :-1
    ]
    errors = policy_errors(policy, pyproject)
    assert any(
        "extra 'interop-all' does not exactly match" in error for error in errors
    )
    assert any("aggregate 'interop-all' does not equal" in error for error in errors)


def test_flagos_and_torch_fl_remain_externally_managed() -> None:
    policy, pyproject = _inputs()
    pyproject["project"]["optional-dependencies"]["flagos"] = ["torch-fl>=1"]
    policy["extras"]["flagos"] = ["torch-fl>=1"]
    policy["classes"]["interop"].append("flagos")
    errors = policy_errors(policy, pyproject)
    assert "FlagOS must not be a FlagQuantum install extra" in errors
    assert "Torch-FL must not be managed by FlagQuantum packaging" in errors


def test_core_import_does_not_load_optional_frameworks() -> None:
    policy, _ = _inputs()
    forbidden = policy["import_policy"]["core_forbidden_imports"]
    code = f"""
import sys
import flagquantum
forbidden = {forbidden!r}
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in forbidden)
)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
