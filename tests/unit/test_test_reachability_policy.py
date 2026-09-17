"""Every test must be reachable by a CI lane that actually exists.

`docs/development/TESTING.md` and `AGENTS.md` define verification as marker
selection, and warn that an empty selection is not evidence. A test with no
marker a lane selects never runs on a pull request at all.

An earlier file-level guard was too weak: a file with one marked test and fifty
unmarked ones passed it. This walks each file's syntax tree instead, so every
test function is checked individually, and it checks the marker against the
selectors the workflows really use rather than merely "not a builtin".

The lane list below mirrors `.github/workflows/ci.yml` and `tools/ci_tier.py`.
Adding a lane means adding its selector here; removing a lane's last selector
means the tests that relied on it fail this check rather than going quiet.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# Selectors any wired lane can select on its own.
#
#   pr-default            smoke, unit
#   pr-runtime            integration
#   pr-distributed        distributed_cpu; benchmark_contract or release_gate
#   coverage              smoke, unit, integration, jax
#   jax-optional          jax
#   qiskit-optional       qiskit
#   pennylane-optional    pennylane
#   multinode-scheduled   distributed_multinode
#
# `braket` has no lane of its own; those tests are reachable because they also
# carry `unit`. The accelerator lane selects `distributed_accel and gpu`, so
# that pair is handled separately below.
LANE_SELECTORS = frozenset(
    {
        "smoke",
        "unit",
        "integration",
        "distributed_cpu",
        "distributed_multinode",
        "benchmark_contract",
        "release_gate",
        "jax",
        "qiskit",
        "pennylane",
        "braket",
    }
)

# The gpu-scheduled lane runs `-m "distributed_accel and gpu"`.
ACCEL_LANE = frozenset({"distributed_accel", "gpu"})

# Marks that shape how a test runs without placing it in a lane.
BUILTIN_MARKS = frozenset(
    {
        "filterwarnings",
        "parametrize",
        "skip",
        "skipif",
        "tryfirst",
        "trylast",
        "usefixtures",
        "xfail",
    }
)


def _mark_names(node: ast.AST) -> set[str]:
    """Marker names referenced by one decorator or assignment value."""
    names: set[str] = set()
    candidates = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    for candidate in candidates:
        if (
            isinstance(candidate, ast.Attribute)
            and isinstance(candidate.value, ast.Attribute)
            and candidate.value.attr == "mark"
        ):
            names.add(candidate.attr)
    return names


def _module_marks(tree: ast.Module) -> set[str]:
    marks: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            marks |= _mark_names(node.value)
    return marks


def _is_test_function(node: ast.AST) -> bool:
    return isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and node.name.startswith("test_")


def _decorator_marks(node: ast.AST) -> set[str]:
    marks: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        marks |= _mark_names(decorator)
    return marks


def _tests(tree: ast.Module) -> list[tuple[str, set[str], int]]:
    """Yield (qualified name, own marks, line) for every test function."""
    found: list[tuple[str, set[str], int]] = []
    for node in tree.body:
        if _is_test_function(node):
            found.append((node.name, _decorator_marks(node), node.lineno))
        elif isinstance(node, ast.ClassDef):
            class_marks = _decorator_marks(node)
            for child in node.body:
                if _is_test_function(child):
                    found.append(
                        (
                            f"{node.name}::{child.name}",
                            class_marks | _decorator_marks(child),
                            child.lineno,
                        )
                    )
    return found


def _test_files() -> list[Path]:
    return sorted((ROOT / "tests").rglob("test_*.py"))


def test_every_test_is_reachable_by_a_ci_lane() -> None:
    unreachable: list[str] = []
    for path in _test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_marks = _module_marks(tree)
        for name, own_marks, line in _tests(tree):
            marks = module_marks | own_marks
            reachable = bool(marks & LANE_SELECTORS) or marks >= ACCEL_LANE
            if not reachable:
                shown = sorted(marks - BUILTIN_MARKS) or ["<none>"]
                unreachable.append(
                    f"{path.relative_to(ROOT)}:{line} {name} (marks={','.join(shown)})"
                )

    assert not unreachable, (
        "these tests carry no marker that any CI lane selects, so they never "
        "run on a pull request: " + "; ".join(unreachable)
    )


def test_lane_selectors_are_registered_markers() -> None:
    """A selector the workflows use must also be a declared marker."""
    registered = set()
    for line in (ROOT / "pytest.ini").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if (
            stripped
            and ":" in stripped
            and not stripped.startswith(("[", "#", "markers"))
        ):
            registered.add(stripped.split(":", 1)[0].strip())
        elif stripped.endswith("="):
            registered.add(stripped.rstrip("=").strip())

    for selector in LANE_SELECTORS | ACCEL_LANE:
        assert selector in registered, f"{selector} is not declared in pytest.ini"
