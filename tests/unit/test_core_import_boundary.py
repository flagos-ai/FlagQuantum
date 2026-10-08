"""No module under ``flagquantum/core`` may import a distribution the policy forbids.

``dependency-policy.toml`` declares ``import_policy.core_forbidden_imports``: a
closed list of distributions Core must never import, because Core is the
vendor-neutral layer and ``core`` declares exactly one dependency (``torch``).
That list is read today by two things, and both of them leave a hole:

* ``tools/check_import_time.py`` imports ``flagquantum`` in a child and reports a
  forbidden name that ended up in ``sys.modules``. It therefore only sees a
  module that the eager path actually reaches, and only at import time. That
  reach is measured rather than assumed, and it is *none*: a child that runs
  ``import flagquantum`` loads no module whose name starts with
  ``flagquantum.core``, so every one of the package's 21 modules is invisible to
  that check.
* ``tools/check_architecture.py`` enforces ``[interop_boundaries]`` in
  ``architecture.toml``, which names five frameworks (``braket``, ``cirq``,
  ``cudaq``, ``pennylane``, ``qiskit``). The other names on the policy list have
  no boundary rule to enforce them: 13 of the 19 declared names have no reader at
  all.

So ``def f(): import jax`` inside ``flagquantum/core``, and even a module-level
``import stim`` in any Core module, passes every gate that runs. This test states
the rule for the whole Core package at every nesting level, and reads the list
from the policy file rather than restating it, so the policy stays the single
source of truth.

The scan is syntactic and covers both directions an import can take:

* ``import jax``, ``import jax.numpy``, ``import jax as jnp`` -- an
  :class:`ast.Import` whose alias names the forbidden distribution or a submodule
  of it. The bound name never decides: ``import jax as anything`` is a violation.
* ``from qiskit import ...``, ``from qiskit.circuit import ...`` -- an
  :class:`ast.ImportFrom` whose module is the forbidden distribution or a
  submodule.

An import of a package that merely shares a prefix -- ``jaxlib`` versus ``jax``,
``stimulus`` versus ``stim`` -- is not a violation, so names are compared on the
dotted boundary rather than by raw string prefix. ``qiskit_aer`` is on the list
as its own entry for the same reason.

The walk is fail-closed about its own coverage: a scan that stopped finding files
would report no violations and look green, so the files it must cover and the
size of the list it enforces are pinned separately below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.check_dependency_policy import load_toml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "dependency-policy.toml"
CORE_PACKAGE = ROOT / "flagquantum" / "core"


def _forbidden_distributions() -> tuple[str, ...]:
    """Return the policy's Core-forbidden distribution names, sorted and unique."""
    policy = load_toml(POLICY)
    declared = policy["import_policy"]["core_forbidden_imports"]
    return tuple(sorted(set(declared)))


def _core_modules() -> list[Path]:
    """Return every module of the Core package, in path order."""
    return sorted(CORE_PACKAGE.rglob("*.py"))


def _is_forbidden(module: str, forbidden: tuple[str, ...]) -> bool:
    """Return whether a dotted module name is a forbidden distribution or inside one."""
    return any(module == name or module.startswith(f"{name}.") for name in forbidden)


def _forbidden_imports(path: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Return a ``line: statement`` description of every forbidden import in ``path``.

    Both statement forms are read at every level of the file, so an import inside a
    function, a class body, a ``try`` block, or a ``TYPE_CHECKING`` block is found
    the same as a top-level one. Nothing is executed: the file is parsed and the
    import nodes inspected.

    Args:
        path: The source file to read.
        forbidden: The distribution names the policy forbids in Core.

    Returns:
        One description per forbidden import, in line order, empty if there is none.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name, forbidden):
                    offenders.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no distribution to resolve, and ``None`` is what
            # a ``from . import x`` gives; neither can name a third-party package.
            module = node.module or ""
            if _is_forbidden(module, forbidden):
                offenders.append((node.lineno, f"from {module} import ..."))
    return [f"{line}: {statement}" for line, statement in sorted(offenders)]


def test_no_core_module_imports_a_policy_forbidden_distribution() -> None:
    """Core is the vendor-neutral layer, and the policy names what it may not import."""
    forbidden = _forbidden_distributions()
    offenders: list[str] = []
    for path in _core_modules():
        for finding in _forbidden_imports(path, forbidden):
            offenders.append(f"{path.relative_to(ROOT)}:{finding}")

    assert not offenders, (
        "dependency-policy.toml declares these distributions forbidden to Core, and "
        "Core owns exactly one declared dependency (torch). Importing one here is a "
        "hidden dependency on an optional framework: the eager import-time check "
        "cannot see it and the interop boundaries name only five frameworks. Move the "
        "dependency to the layer that owns it: " + "; ".join(offenders)
    )


def test_the_guard_scans_the_whole_core_package() -> None:
    """A scan that found nothing would pass the check above without checking anything.

    The files and the rule set this guard reads are pinned here, so a moved package,
    a glob that stopped matching, or a policy list that shrank to nothing fails this
    test instead of quietly turning the guard above green.
    """
    modules = {path.relative_to(ROOT).as_posix() for path in _core_modules()}

    assert "flagquantum/core/__init__.py" in modules
    assert "flagquantum/core/operator_schema.py" in modules
    assert "flagquantum/core/ir/__init__.py" in modules
    assert "flagquantum/core/ir/program/verifier.py" in modules
    # The walk is recursive, so a module in a subdirectory added later is covered.
    assert len(modules) >= 15, sorted(modules)

    forbidden = _forbidden_distributions()
    assert len(forbidden) >= 15, forbidden
    # One member per enforcement shape: a framework with its own interop boundary, a
    # kernel accelerator, and a stabilizer decoder, so a shrunken list is visible.
    assert "qiskit" in forbidden
    assert "jax" in forbidden
    assert "stim" in forbidden


def test_the_forbidden_names_are_matched_on_the_dotted_boundary() -> None:
    """The distribution name decides, not the alias and not a lookalike prefix.

    ``jax`` and ``jax.numpy`` are violations; ``jaxlib`` is not, and is a violation
    only because the policy lists it separately.
    """
    assert _is_forbidden("jax", ("jax", "jaxlib"))
    assert _is_forbidden("jax.numpy", ("jax", "jaxlib"))
    assert _is_forbidden("jaxlib", ("jax", "jaxlib"))
    assert not _is_forbidden("jaxlib_extra", ("jax",))
    assert not _is_forbidden("jaxx", ("jax",))
    assert not _is_forbidden("", ("jax",))


def test_both_import_forms_are_read_at_every_nesting_level(tmp_path: Path) -> None:
    """The reader is exercised on a source written to say what it has to find.

    The scan above runs against Core as it is today, and Core is clean, so every
    branch of the reader reports nothing there. A branch with no positive witness
    is a branch a later edit can break while the suite stays green, so this test
    states each one on a file that carries it: an import inside a function, an
    aliased import, a module-level import, and an ``ImportFrom`` -- the only branch
    that reads ``node.module`` rather than an alias -- beside the two shapes that
    must *not* be reported, a relative import and a lookalike prefix.

    The nested import is written *above* the module-level ones on purpose. The walk
    reaches the module's own statements before the ones nested inside its functions,
    so the findings arrive out of line order and the reported list is only in line
    order because the reader sorts it.
    """
    source = tmp_path / "probe.py"
    source.write_text(
        "def inner():\n"
        "    import stim as sampler\n"
        "    from . import sibling\n"
        "    from .program import model\n"
        "    return (sampler, sibling, model)\n"
        "\n"
        "\n"
        "import jax\n"
        "import jaxlib_extra\n"
        "from matplotlib import pyplot\n",
        encoding="utf-8",
    )

    assert _forbidden_imports(source, ("jax", "stim", "matplotlib")) == [
        "2: import stim",
        "8: import jax",
        "10: from matplotlib import ...",
    ]
