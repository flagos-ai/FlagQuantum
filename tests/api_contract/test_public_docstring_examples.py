"""Public workflow examples remain executable and free of external effects."""

import ast
import doctest
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.ecosystem.qiskit import run as run_qiskit
from flagquantum.runtime import planner

pytestmark = pytest.mark.unit

# Every entry whose docstrings carry examples. `fq.plan` and `planner.plan` are
# different functions that document different things, so both are listed.
ENTRIES = (
    fq.Circuit,
    fq.Module,
    fq.Observable,
    fq.compile,
    fq.expectation,
    fq.plan,
    fq.run,
    fq.train,
    planner.plan,
    run_qiskit,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _modules_with_examples() -> set[str]:
    """Return the importable module names whose docstrings carry an example."""
    modules = set()
    for path in sorted((_REPOSITORY_ROOT / "flagquantum").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            docstring = ast.get_docstring(node)
            if docstring and ">>>" in docstring:
                relative = path.relative_to(_REPOSITORY_ROOT).with_suffix("")
                parts = list(relative.parts)
                if parts[-1] == "__init__":
                    parts.pop()
                modules.add(".".join(parts))
                break
    return modules


def test_primary_public_docstring_examples() -> None:
    runner = doctest.DocTestRunner()
    finder = doctest.DocTestFinder()

    for entry in ENTRIES:
        name = f"{entry.__module__}.{entry.__name__}"
        for example in finder.find(entry, name=name):
            runner.run(example)

    failures, _ = runner.summarize()
    assert failures == 0


def test_every_module_with_examples_is_covered() -> None:
    """Fail when an example appears in a module this gate does not run.

    The list of entries above is what makes an example executable, so a
    docstring written in a module outside that list would look tested while
    nothing ran it. Adding the module's public entry to ``ENTRIES`` is the fix.
    """
    covered = {entry.__module__ for entry in ENTRIES}
    uncovered = sorted(_modules_with_examples() - covered)

    assert uncovered == [], (
        "docstring examples in modules that no doctest entry runs: " f"{uncovered}"
    )
