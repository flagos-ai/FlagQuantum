"""Every name a benchmark imports from this package has to exist.

`benchmarks/` is not linted and nothing imports it, so a name that moves out
from under a script is invisible until a lane runs that script -- and then the
failure is an `ImportError` at the first line, which says nothing about the
refactor that caused it. That is not hypothetical: three scripts were importing
names their modules no longer export, one of them in the weekly hardware lane,
where the phase had never once got past its own imports.

The check resolves each `from flagquantum... import X` against the module it
names, using the module itself. It does not import the benchmark: these scripts
need accelerators, optional extras and a distributed launch, and a guard that
needed all of that would not run on the lane that has to fail.

Not every reference can be answered for here, and that is counted rather than
ignored -- a module behind an optional extra that is not installed is not
evidence either way, so the number of references actually checked has to stay
high enough that the guard cannot quietly go blind. `hasattr` is a trap for
this: a PEP 562 `__getattr__` that ends in `ImportError` propagates out of
`hasattr`, which only swallows `AttributeError`.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
PACKAGE = "flagquantum"

# Below these, the check is reading a tree it cannot see rather than a tree with
# nothing wrong, and saying so is the point of counting.
MIN_RESOLVED_MODULES = 20
MIN_CHECKED_REFERENCES = 25

# `(file, line, module, name, present)` for a reference this environment could
# answer for.
Checked = tuple[str, int, str, str, bool]


def _references() -> list[tuple[str, int, str, str]]:
    """Every `from flagquantum... import X` in `benchmarks/`.

    Imports inside functions are walked too: a deferred import is still a
    reference that has to resolve the first time that path runs.
    """
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(BENCHMARKS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level or not node.module:
                continue
            if node.module != PACKAGE and not node.module.startswith(f"{PACKAGE}."):
                continue
            for alias in node.names:
                found.append((path.name, node.lineno, node.module, alias.name))
    return found


def _is_our_own_missing_path(error: BaseException) -> bool:
    """A `flagquantum` path that does not exist is a defect, not an absent extra.

    The distinction is the whole reason this is a function: `importlib` answers
    "cannot import" the same way whether a third-party extra is uninstalled or
    the module a benchmark names has been moved. Only the second is a defect,
    and the error says which -- `ModuleNotFoundError.name` is the package that
    was not found. Without this, a moved module is silently filed under "cannot
    tell" and the guard reports nothing, which is exactly what it did the first
    time this check was written.
    """
    return isinstance(error, ModuleNotFoundError) and str(error.name or "").startswith(
        PACKAGE
    )


def _resolve() -> tuple[list[Checked], list[str], list[str]]:
    references = _references()
    assert references, "no benchmark imports this package at all; the scan is blind"
    assert len({file for file, _, _, _ in references}) >= 10, sorted(
        {file for file, _, _, _ in references}
    )

    modules: dict[str, object] = {}
    checked: list[Checked] = []
    unanswerable: list[str] = []
    moved: list[str] = []
    for file, line, module, name in references:
        if module not in modules:
            try:
                modules[module] = importlib.import_module(module)
            except Exception as error:  # classified below
                modules[module] = None
                if _is_our_own_missing_path(error):
                    moved.append(module)
                else:
                    unanswerable.append(module)
        target = modules[module]
        if target is None:
            continue
        try:
            present = hasattr(target, name)
        except Exception:  # a lazy re-export behind an absent extra
            unanswerable.append(f"{module}.{name}")
            continue
        checked.append((file, line, module, name, present))
    return checked, unanswerable, moved


@pytest.fixture(scope="module")
def resolved() -> tuple[list[Checked], list[str], list[str]]:
    return _resolve()


def test_the_scan_resolves_enough_of_the_tree_to_speak_for_it(resolved) -> None:
    checked, unanswerable, _moved = resolved
    modules = {module for _, _, module, _, _ in checked}

    assert len(modules) >= MIN_RESOLVED_MODULES, (
        f"only {len(modules)} modules could be reached, so the name check below "
        f"cannot speak for the rest: {sorted(unanswerable)}"
    )
    assert len(checked) >= MIN_CHECKED_REFERENCES, (
        f"only {len(checked)} benchmark imports could be checked, which is too "
        f"few to speak for the rest: {sorted(unanswerable)}"
    )


def test_no_benchmark_imports_a_module_that_no_longer_exists(resolved) -> None:
    _checked, _unanswerable, moved = resolved

    assert not moved, (
        "these benchmarks import from package paths that do not exist at all, "
        "so the script fails at its own import line:\n" + "\n".join(sorted(moved))
    )


def test_a_moved_module_is_reported_rather_than_excused() -> None:
    """The guard has to tell a moved module from an uninstalled extra.

    Both read as "cannot import". If the first is filed with the second, the
    check goes quiet on the one case it exists to catch -- which is what the
    first version of this file did.
    """

    def raise_for(name: str) -> ModuleNotFoundError:
        error = ModuleNotFoundError(f"No module named {name!r}")
        error.name = name
        return error

    assert _is_our_own_missing_path(raise_for("flagquantum.simulation.dense_island"))
    assert _is_our_own_missing_path(raise_for("flagquantum.runtime.executors.mps.gone"))
    assert not _is_our_own_missing_path(raise_for("triton"))
    assert not _is_our_own_missing_path(raise_for("jax.numpy"))
    # A lazy re-export that fails inside our package is still an absent extra,
    # not a moved path, so ImportError and its subclasses that do not name a
    # missing module are left alone.
    assert not _is_our_own_missing_path(ImportError("no triton"))


def test_every_name_a_benchmark_imports_still_exists(resolved) -> None:
    checked, _unanswerable, _moved = resolved
    missing = [
        f"{file}:{line} {module}.{name}"
        for file, line, module, name, present in checked
        if not present
    ]

    assert not missing, (
        "these benchmarks import names their modules no longer export, so the "
        "script fails at its own import line before it measures anything:\n"
        + "\n".join(missing)
    )
