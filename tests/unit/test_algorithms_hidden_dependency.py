"""No algorithms surface may reach for a dependency this repository does not declare.

``numpy`` is not a dependency of FlagQuantum and is not on the list of imports the
core dependency policy forbids, so nothing fails when a module or a test in the
algorithms package imports it. The algorithms test files currently pass with
numpy importable and would pass without it, which is exactly why the boundary has
to be stated rather than inferred: a lane that never resolves the import cannot
report it. This test states it.

The scan is syntactic and covers both directions an import can take, over the
three forms that matter:

* ``import numpy`` and ``import numpy as np`` -- an :class:`ast.Import` whose
  alias names ``numpy`` or a submodule of it (``import numpy.linalg``);
* ``from numpy import ...`` and ``from numpy.linalg import ...`` -- an
  :class:`ast.ImportFrom` whose module is ``numpy`` or a submodule.

The bound name never decides: ``import numpy as anything`` is a violation, which
is why the check reads the module and not the alias.

An import of a package that merely shares a prefix -- ``numpyro``,
``numpy_thing`` -- is not a violation, so the check compares on the dotted
boundary rather than on a raw string prefix.

The walk is fail-closed about its own coverage as well: a scan that stopped
finding files would report no violations and look green, so the files it must
cover are pinned separately below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# The package whose modules must stay torch-only, and the test files that
# exercise it. Both are walked; neither may import numpy.
ALGORITHMS_PACKAGE = ROOT / "flagquantum" / "algorithms"
ALGORITHMS_TEST_GLOB = "test_algorithms_*.py"

FORBIDDEN_MODULE = "numpy"


def _algorithm_modules() -> list[Path]:
    """Return every module of the algorithms package, in path order."""
    return sorted(ALGORITHMS_PACKAGE.rglob("*.py"))


def _algorithm_tests() -> list[Path]:
    """Return every algorithms unit test file, in path order."""
    return sorted((ROOT / "tests" / "unit").glob(ALGORITHMS_TEST_GLOB))


def _scanned_files() -> list[Path]:
    """Return every file this guard reads."""
    return _algorithm_modules() + _algorithm_tests()


def _is_forbidden(module: str) -> bool:
    """Return whether a dotted module name is the forbidden module or inside it."""
    return module == FORBIDDEN_MODULE or module.startswith(f"{FORBIDDEN_MODULE}.")


def _forbidden_imports(path: Path) -> list[str]:
    """Return a ``line: statement`` description of every numpy import in ``path``.

    Both statement forms are read at every level of the file, so an import inside
    a function, a class body, or a ``try`` block is found the same as a top-level
    one. Nothing is executed: the file is parsed and the import nodes inspected.

    Args:
        path: The source file to read.

    Returns:
        One description per forbidden import, in line order, empty if there is none.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    offenders.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module to resolve, and ``None`` is what an
            # ``from . import x`` gives; neither can name numpy.
            module = node.module or ""
            if _is_forbidden(module):
                offenders.append((node.lineno, f"from {module} import ..."))
    return [f"{line}: {statement}" for line, statement in sorted(offenders)]


def test_no_algorithms_module_or_test_imports_numpy() -> None:
    """The algorithms package and its tests are torch-only, by policy."""
    offenders: list[str] = []
    for path in _scanned_files():
        for finding in _forbidden_imports(path):
            offenders.append(f"{path.relative_to(ROOT)}:{finding}")

    assert not offenders, (
        f"{FORBIDDEN_MODULE} is not a declared dependency of this repository, so an "
        "import of it in the algorithms surface is a hidden dependency: no CI lane "
        "installs it on purpose and nothing else here would report it. Import from "
        "torch instead: " + "; ".join(offenders)
    )


def test_the_guard_scans_the_whole_algorithms_surface() -> None:
    """A scan that found nothing would pass the check above without checking anything.

    The files this guard must read are pinned here, so a moved package, a renamed
    test file, or a glob that stopped matching fails this test instead of quietly
    turning the guard above green.
    """
    modules = {path.relative_to(ROOT).as_posix() for path in _algorithm_modules()}
    tests = {path.relative_to(ROOT).as_posix() for path in _algorithm_tests()}

    assert "flagquantum/algorithms/__init__.py" in modules
    assert "flagquantum/algorithms/primitives/__init__.py" in modules
    # Every module the package currently ships, including ones added later: the
    # scan is recursive, so a module in a subdirectory is covered too.
    assert len(modules) >= 10, sorted(modules)
    assert "tests/unit/test_algorithms_package.py" in tests, sorted(tests)
    assert len(tests) >= 8, sorted(tests)


def test_the_forbidden_module_is_matched_on_the_dotted_boundary() -> None:
    """The module name decides, not the alias and not a lookalike prefix.

    ``numpy`` and ``numpy.linalg`` are violations; a package whose name merely
    starts with the same letters is not.
    """
    assert _is_forbidden("numpy")
    assert _is_forbidden("numpy.linalg")
    assert not _is_forbidden("numpyro")
    assert not _is_forbidden("numpy_helpers")
    assert not _is_forbidden("")
