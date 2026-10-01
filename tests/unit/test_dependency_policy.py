from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.check_dependency_policy import (
    development_floor_pins,
    load_toml,
    policy_errors,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
# The static gates whose declared floor the `dependency-bounds` lane installs and
# runs, so a floor that cannot pass its own check fails CI instead of shipping.
GATED_TOOLS = ("black", "mypy", "ruff")


def _inputs() -> tuple[dict, dict]:
    return (
        load_toml(ROOT / "dependency-policy.toml"),
        load_toml(ROOT / "pyproject.toml"),
    )


def _without_upper_bound(extras: list[str], name: str) -> list[str]:
    return [
        (
            requirement.split(",")[0]
            if requirement.startswith(f"{name}>=")
            else requirement
        )
        for requirement in extras
    ]


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


def test_development_requirements_must_declare_an_upper_bound() -> None:
    """An unbounded development entry lets upstream change this toolchain silently.

    `dev` is what the repository installs to check itself, so a requirement with
    no upper bound can break `main` with no commit and no diff to review.
    """
    policy, pyproject = _inputs()
    for extras in (policy["extras"], pyproject["project"]["optional-dependencies"]):
        extras["dev"] = _without_upper_bound(extras["dev"], "pytest")
    errors = policy_errors(policy, pyproject)
    assert (
        "development requirement 'pytest>=7.0.0' must declare an upper bound" in errors
    )


def test_recorded_versions_must_match_the_declared_range() -> None:
    policy, pyproject = _inputs()
    policy["tested"]["ruff"] = ["0.1.0", "0.16.9"]
    errors = policy_errors(policy, pyproject)
    assert any(
        "recorded dependency 'ruff' starts at '0.1.0'" in error for error in errors
    )

    policy, pyproject = _inputs()
    policy["tested"]["ruff"] = ["0.15.0", "0.17.0"]
    errors = policy_errors(policy, pyproject)
    assert any(
        "recorded dependency 'ruff' records '0.17.0'" in error for error in errors
    )


def test_recorded_dependencies_must_be_declared_by_the_policy() -> None:
    policy, pyproject = _inputs()
    policy["tested"]["phantom-sdk"] = ["1.0"]
    errors = policy_errors(policy, pyproject)
    assert "recorded dependency 'phantom-sdk' is not declared" in errors


def test_an_exact_pin_must_record_the_pinned_release() -> None:
    """An exact pin is an upper bound, but only one release may be recorded."""
    policy, pyproject = _inputs()
    for extras in (policy["extras"], pyproject["project"]["optional-dependencies"]):
        for name, requirements in extras.items():
            extras[name] = [
                "ruff==0.15.0" if item.startswith("ruff>=") else item
                for item in requirements
            ]
    policy["tested"]["ruff"] = ["0.15.0"]
    assert policy_errors(policy, pyproject) == ()

    policy["tested"]["ruff"] = ["0.15.0", "0.16.9"]
    errors = policy_errors(policy, pyproject)
    assert (
        "recorded dependency 'ruff' is pinned to '0.15.0' but records '0.16.9'"
        in errors
    )


def test_development_floor_pins_are_the_recorded_static_gate_floors() -> None:
    policy, _ = _inputs()
    pinned = dict(pin.split("==") for pin in development_floor_pins(policy))
    assert set(pinned) == set(GATED_TOOLS)
    for tool in GATED_TOOLS:
        assert pinned[tool] == policy["tested"][tool][0]


def test_the_bounds_lane_installs_and_runs_the_declared_floor() -> None:
    """The declared floor is a claim about this toolchain, so CI must exercise it.

    A recorded floor that no lane installs is a number in a file: `mypy>=1.0`
    was declared while the current release was the only one ever installed, and
    1.0.0 cannot type-check this repository at all.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["dependency-bounds"]

    install = [
        step for step in job["steps"] if step.get("if") == "matrix.python == '3.12'"
    ]
    assert [step["name"] for step in install] == [
        "Install the declared development floor",
        "Run the Python gates on the declared development floor",
    ], "the 3.12 lane should pin the declared floor and then run the gates"
    assert "--print-floor" in install[0]["run"], (
        "the floor lane must install the pins the policy records, not a second "
        "hand-maintained list"
    )
    for gate in GATED_TOOLS:
        assert gate in install[1]["run"], f"the floor lane never runs {gate}"
