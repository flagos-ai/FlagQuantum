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
    reference_errors,
    stdlib_fallback_errors,
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


def _sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, str]):
    """Point the scans at a written-out tree instead of the repository."""

    from tools import check_dependency_policy as gate

    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    written = tuple(sorted(tmp_path.rglob("*.py")))
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "python_files", lambda trees: written)
    return gate


def test_collected_modules_guard_every_optional_reference() -> None:
    """An optional root may appear in a collected module only behind a guard.

    An environment installed from `.[dev]` has no optional extra, so an
    unguarded reference is a collection error on that lane and a green light
    everywhere else. `tests/unit/test_circuit_power.py` shipped exactly that
    with numpy and the density-matrix contract suite shipped it with `tomllib`.
    """

    policy, _ = _inputs()
    assert reference_errors(policy) == ()
    assert stdlib_fallback_errors() == ()


def test_the_reference_scan_covers_every_collected_tree() -> None:
    """A scan that silently reads nothing must not report success."""

    from tools.check_dependency_policy import COLLECTION_TREES, python_files

    scanned = {
        path.relative_to(ROOT).parts[0] for path in python_files(COLLECTION_TREES)
    }
    assert scanned == set(COLLECTION_TREES)
    assert len(python_files(COLLECTION_TREES)) > 500


def test_an_unguarded_optional_import_in_a_collected_module_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = _sandbox(
        monkeypatch,
        tmp_path,
        {
            "tests/unit/test_example.py": "import numpy as np\n\n\ndef test_x() -> None:\n    assert np\n"
        },
    )
    policy, _ = _inputs()
    errors = gate.reference_errors(policy)
    assert len(errors) == 1
    assert "tests/unit/test_example.py:1" in errors[0]
    assert "'numpy' is referenced without a guard" in errors[0]


@pytest.mark.parametrize(
    "body",
    [
        'import pytest\n\nstim = pytest.importorskip("stim")\nimport numpy as np  # noqa: E402\n',
        (
            "import importlib.util\n\nimport pytest\n\n"
            'if importlib.util.find_spec("stim") is None:  # pragma: no cover\n'
            '    pytest.skip("stim is not installed", allow_module_level=True)\n'
            "import numpy as np  # noqa: E402\n"
        ),
        (
            "import pytest\n\n"
            'if pytest.importorskip("stim") is None:  # pragma: no cover\n'
            "    raise SystemExit(1)\n"
            "import numpy as np  # noqa: E402\n"
        ),
        "try:\n    import numpy as np\nexcept ModuleNotFoundError:\n    np = None\n",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import numpy as np\n",
    ],
)
def test_a_guard_that_holds_when_the_root_is_absent_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    """The accepted forms, including a guard on a dependency that requires it."""

    gate = _sandbox(
        monkeypatch,
        tmp_path,
        {"tests/unit/test_example.py": body + "\n\ndef test_x() -> None:\n    pass\n"},
    )
    policy, _ = _inputs()
    assert gate.reference_errors(policy) == ()


def test_inspecting_a_specification_is_not_a_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`find_spec` only gates the module when the result also skips it."""

    gate = _sandbox(
        monkeypatch,
        tmp_path,
        {
            "tests/unit/test_example.py": (
                "import importlib.util\n\n"
                'if importlib.util.find_spec("stim") is not None:\n'
                "    pass\n"
                "import numpy as np\n"
            )
        },
    )
    policy, _ = _inputs()
    assert len(gate.reference_errors(policy)) == 1


def test_a_standard_library_import_must_carry_its_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`tomllib` is a 3.11 addition, so a version test is not a fallback."""

    gate = _sandbox(
        monkeypatch,
        tmp_path,
        {
            "tools/example.py": (
                "import sys\n\n"
                "if sys.version_info >= (3, 11):\n"
                "    import tomllib\n"
                "else:  # pragma: no cover - Python 3.10\n"
                "    import tomli as tomllib\n\n"
                "print(tomllib)\n"
            )
        },
    )
    errors = gate.stdlib_fallback_errors()
    assert len(errors) == 1
    assert "tools/example.py:4" in errors[0]
    assert "needs the fallback form" in errors[0]


def test_guard_implications_must_name_forbidden_roots() -> None:
    """The implication table cannot quietly exempt a root from the scan."""

    policy, pyproject = _inputs()
    policy["import_policy"]["guard_implications"]["numpy"] = ["stim", "scipy"]
    errors = policy_errors(policy, pyproject)
    assert any("guard_implications['numpy'] names 'scipy'" in error for error in errors)
    policy["import_policy"]["guard_implications"]["numpy"] = ["numpy"]
    errors = policy_errors(policy, pyproject)
    assert any(
        "guard_implications['numpy'] cannot name itself" in error for error in errors
    )


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
