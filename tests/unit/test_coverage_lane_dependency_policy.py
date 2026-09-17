"""The coverage lane must install what the tests its selector reaches depend on.

`.github/workflows/ci.yml` runs the coverage job with a marker expression and a
fixed install line, and the two are maintained separately. `integration` is a
broad selector: it also reaches the optional-integration suites. When the
install line does not carry the package one of those suites needs, the failure
is silent -- the suite calls `pytest.importorskip`, pytest records a skip, and a
skip is indistinguishable from a pass in the summary. The module the suite
exercises is never imported, so it measures as untested and no lane goes red.

`flagquantum/ecosystem/pennylane` was in exactly that state: its suite carried
`integration`, so the coverage lane selected it, but the lane installed only
`.[dev,jax,viz]`, so every one of its tests skipped and the package sat below
half covered with nothing failing.

This checks the two halves against each other. A test the coverage lane selects
must not reach for an optional integration the lane neither installs nor
excludes: add the extra, or take the marker out of the selector, but decide
explicitly rather than letting the skip absorb the difference.

`qiskit` takes the second route. Its native libraries cannot be loaded once the
rest of the tree has been imported -- `qiskit/_accelerate.abi3.so` raises
`ImportError: cannot allocate memory in static TLS block`, and loading that one
first only moves the failure to `qiskit_aer.libs/libgomp-*.so`. pytest reports
it as a collection error that aborts the lane outright, which is worse than the
skip it replaces. Those tests still run in the `qiskit-optional` job, which
installs only `.[dev]` and imports them without the rest of the tree.

The expression only reaches the tests that probe Qiskit inside the test body.
Three files gate the whole module on `importorskip`, which fires while the
module is imported, before the marker filter is applied; each of those still
records one module-level skip. That skip names its cause, so it hides nothing.
A skip only becomes a hole when it is the reason a package measures far below
what its suite actually covers, which is what pennylane was.

Scope: this reasons about optional-integration extras, which a lane can install.
A test that skips for want of a device or an unset environment variable is a
different case -- the lane cannot install its way out of it -- and this check
does not speak to those.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# The integrations a test can ask for at run time, keyed by the import name it
# probes. `jaxlib` and `qiskit_aer` ship inside another extra, so they name the
# extra that carries them rather than one of their own.
PROBE_EXTRA = {
    "jax": "jax",
    "jaxlib": "jax",
    "pennylane": "pennylane",
    "qiskit": "qiskit",
    "qiskit_aer": "qiskit",
    "braket": "braket",
}

# The import-name lookups that make a test, or a whole module, skip.
PROBE = re.compile(r"""(?:importorskip|find_spec)\(\s*["']([A-Za-z_][\w.]*)["']""")

TOKEN = re.compile(r"[A-Za-z_]\w*")


def _coverage_job() -> str:
    """The text of the `coverage:` job, from its header to the next job."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.rstrip() == "  coverage:"
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.match(r"^  \S", lines[index])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _coverage_lane() -> tuple[set[str], set[str], set[str]]:
    """The coverage job's (installed extras, selectors, exclusions)."""
    job = _coverage_job()

    install = next(
        line for line in job.splitlines() if re.search(r"-e ['\"]\.\[", line)
    )
    extras = {
        extra.strip()
        for extra in re.search(r"\.\[([^\]]*)\]", install).group(1).split(",")
        if extra.strip()
    }

    expression = re.search(r'-m\s+"([^"]+)"', job)
    assert expression, "the coverage job should select tests with -m"
    selectors: set[str] = set()
    exclusions: set[str] = set()
    negated = False
    for token in TOKEN.findall(expression.group(1)):
        if token in {"and", "or"}:
            negated = False
        elif token == "not":
            negated = True
        else:
            (exclusions if negated else selectors).add(token)

    return extras, selectors, exclusions


def _marks(node: ast.AST) -> set[str]:
    """Marker names on one decorator or one `pytestmark` value."""
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
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            found |= _marks(node.value)
    return found


def _decorator_marks(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        found |= _marks(decorator)
    return found


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.name.startswith("test_")
    )


def _probes(text: str) -> set[str]:
    return {PROBE_EXTRA.get(name, name) for name in PROBE.findall(text)} & set(
        PROBE_EXTRA.values()
    )


def _lines(source: list[str], node: ast.AST) -> str:
    return "\n".join(source[node.lineno - 1 : node.end_lineno])


def _module_scope(tree: ast.Module, source: list[str]) -> str:
    """Source outside every test function: what a module-level skip depends on."""
    return "\n".join(
        _lines(source, node)
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _tests(
    tree: ast.Module, source: list[str]
) -> list[tuple[str, int, set[str], set[str]]]:
    """(name, line, marks, probes) for every test the coverage lane could reach."""
    found: list[tuple[str, int, set[str], set[str]]] = []
    for node in tree.body:
        if _is_test(node):
            found.append(
                (
                    node.name,
                    node.lineno,
                    _decorator_marks(node),
                    _probes(_lines(source, node)),
                )
            )
        elif isinstance(node, ast.ClassDef):
            class_marks = _decorator_marks(node)
            for child in node.body:
                if _is_test(child):
                    found.append(
                        (
                            f"{node.name}::{child.name}",
                            child.lineno,
                            class_marks | _decorator_marks(child),
                            _probes(_lines(source, child)),
                        )
                    )
    return found


def test_the_coverage_lane_installs_every_integration_it_selects() -> None:
    extras, selectors, exclusions = _coverage_lane()

    offenders: list[str] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        source = path.read_text(encoding="utf-8").splitlines()
        tree = ast.parse("\n".join(source), filename=str(path))
        module_marks = _module_marks(tree)
        module_probes = _probes(_module_scope(tree, source))

        for name, line, own_marks, own_probes in _tests(tree, source):
            marks = module_marks | own_marks
            if not marks & selectors or marks & exclusions:
                continue
            unmet = sorted((module_probes | own_probes) - extras)
            if unmet:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line} {name} "
                    f"(needs {','.join(unmet)})"
                )

    assert not offenders, (
        "the coverage lane selects these tests, but the coverage job does not "
        "install what they need, so they skip and the code they cover is never "
        "measured: " + "; ".join(offenders)
    )


def test_the_coverage_lane_has_a_single_install_and_selector() -> None:
    """The parser above reads one line of each; keep it that way."""
    job = _coverage_job()
    installs = [line for line in job.splitlines() if re.search(r"-e ['\"]\.\[", line)]
    commands = [line for line in job.splitlines() if "-m " in line and "pytest" in line]
    assert len(installs) == 1, installs
    assert len(commands) == 1, commands
