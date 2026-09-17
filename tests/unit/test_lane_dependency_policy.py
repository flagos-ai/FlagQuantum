"""Lane dependency policy: can any lane actually run the test it selects?

The workflows select tests by marker and install dependencies separately. A test
that calls `pytest.importorskip` records a skip when its package is missing, and
pytest reports a skip the same way it reports a pass. So a lane can select a test
it cannot run and still come up green. Two failures of that shape were found
here:

- `flagquantum/ecosystem/pennylane` measured 43.4%. The coverage job selected its
  suite -- `integration` reaches it -- but installed only `.[dev,jax,viz]`, so
  every test skipped and the package measured as barely covered with nothing
  failing. That job now installs the extra.
- Six Triton test files under `tests/unit` gated themselves on the triton extra
  while carrying only `unit`, so every lane selected them and every lane skipped
  them. Seventeen tests executed in no lane at all. They now carry a `triton`
  marker: `triton-optional` runs the four that need no device, and the
  accelerator tier runs the thirteen that do.

The two invariants below cover the two halves. The first is narrower and
stronger: a lane whose job is to measure must run what it selects, or its
measurement is a lie. The second is broader and weaker: every optional
integration a test asks for must be installed by *some* lane that selects that
test and, when the test is marked for hardware, has the hardware.

Scope: both reason about optional-integration extras, which a lane can install.
What a lane runs is read from the `-m` expressions in the workflows and from the
`tools/ci_tier.py` tiers they invoke, so a job whose selection lives in a tier is
seen too. A test that skips for an unset environment variable is outside both:
no install line can fix it, and no lane is configured to set it. So is a probe
for a package no extra declares -- the fix there is a dependency decision, not a
lane wiring one -- and `tests/test_tensor_network.py` has one, for `cotengra`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from tools.check_dependency_policy import distribution_name, load_toml
from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
CI_WORKFLOW = "ci.yml"
POLICY = ROOT / "dependency-policy.toml"

# The policy names distributions, the tests probe import names, and the two
# usually agree once both are normalized. These are the places they do not: a
# probe for the import name has to resolve to the distribution that provides it,
# or the test would go unchecked rather than reported. The sanity test below
# fails if any entry stops naming a declared distribution, so a rename shows up
# as a failure instead of as silence.
IMPORT_ALIASES = {
    "braket": "amazon-braket-sdk",
    "jaxlib": "jax",
    "quark": "quarkcircuit",
}

IMPORTORSKIP = frozenset({"importorskip"})
# `find_spec` only makes a test skip when it is the condition of a skip mark; the
# same call used to guard a conditional import does not.
SKIP_CONDITION = frozenset({"importorskip", "find_spec"})

# Markers that mean the test needs hardware, not just a package.
DEVICE_MARKS = frozenset({"gpu", "distributed_accel"})
# The runner labels a self-hosted accelerator job uses.
DEVICE_LABEL = re.compile(r"\bruns-on:.*\bgpu\b|\bruns-on:.*\baccelerator\b")

TOKEN = re.compile(r"[A-Za-z_]\w*")
JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
NEW_KEY = re.compile(r"[A-Za-z_][\w-]*:")
EXTRAS = re.compile(r"-e\s+['\"]\.\[([^\]]*)\]")
QUOTED_EXPRESSION = re.compile(r'-m\s+"([^"]+)"')
BARE_EXPRESSION = re.compile(r"-m\s+([A-Za-z_]\w*)")
DISTRIBUTION = re.compile(r"[\w-]+")
TEST_FILE = re.compile(r"(tests/[\w./-]+\.py)")
TIER = re.compile(r"ci_tier\.py\s+([\w-]+)")
# A pytest invocation, not any mention of the name. `import pytest, torch` is not
# a run, and reading it as one turns the `-m` of the `pip install` that follows it
# in the same shell block into a marker expression called `pip`.
PYTEST = re.compile(r"(?:^|[;&|]|--|python[\d.]*\s+-m)\s*pytest\b")


def _distributions(requirements: list[str]) -> set[str]:
    """The normalized distribution name of each requirement."""
    return {distribution_name(requirement) for requirement in requirements}


def _declared() -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """(distribution -> extras providing it, the core distributions).

    Aggregate extras (`all`, `interop-all`) are skipped: naming one would make a
    lane that installs `jax` look like a lane that installs everything.
    """
    policy = load_toml(POLICY)
    aggregates = set(policy["aggregates"])
    providers: dict[str, set[str]] = {}
    for extra, requirements in policy["extras"].items():
        if extra in aggregates:
            continue
        for distribution in _distributions(requirements):
            providers.setdefault(distribution, set()).add(extra)
    return (
        {name: frozenset(extras) for name, extras in providers.items()},
        frozenset(_distributions(policy["core"])),
    )


PROVIDERS, CORE = _declared()


def extras_for(name: str) -> frozenset[str]:
    """The extras that can provide this import name; empty if nothing declares it."""
    normalized = distribution_name(IMPORT_ALIASES.get(name, name))
    if normalized in CORE:
        return frozenset()
    return PROVIDERS.get(normalized, frozenset())


class Job(NamedTuple):
    """One workflow job: what it installs, and which tests it would run."""

    workflow: str
    name: str
    extras: frozenset[str]
    expressions: tuple[str, ...]
    paths: frozenset[str]
    device: bool

    def selects(self, marks: set[str], path: str) -> bool:
        if path in self.paths:
            return True
        return any(_true_for(expression, marks) for expression in self.expressions)

    def runs(self, marks: set[str], path: str) -> bool:
        """Like `selects`, but a device-marked test also needs the hardware."""
        if not self.selects(marks, path):
            return False
        return self.device or not (marks & DEVICE_MARKS)


def _logical_lines(text: str) -> list[str]:
    """YAML folded scalars joined back into one line each.

    An install command broken across lines by `>-` hides its distributions from
    a line-by-line reader, which is how a job that installs Qiskit looked like a
    job that does not.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        starts_entry = stripped.startswith("- ") or NEW_KEY.match(stripped)
        if starts_entry or not lines:
            lines.append(stripped)
        else:
            lines[-1] += " " + stripped
    return lines


def _job_blocks() -> list[tuple[str, str, str]]:
    """(workflow file name, job name, job text) for every job in every workflow."""
    blocks: list[tuple[str, str, str]] = []
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        lines = workflow.read_text(encoding="utf-8").splitlines()
        if "jobs:" not in lines:
            continue
        start = lines.index("jobs:")
        headers = [
            (index, match.group(1))
            for index, line in enumerate(lines[start + 1 :], start=start + 1)
            if (match := JOB_HEADER.match(line))
        ]
        for position, (index, name) in enumerate(headers):
            end = (
                headers[position + 1][0] if position + 1 < len(headers) else len(lines)
            )
            blocks.append((workflow.name, name, "\n".join(lines[index:end])))
    return blocks


def _tier_expressions(text: str) -> list[str]:
    """The `-m` expressions of every `tools/ci_tier.py <tier>` this text invokes.

    Read from the tier's command tuples rather than from their rendered text,
    which joins the arguments and loses the quoting around an expression.
    """
    found: list[str] = []
    for line in _logical_lines(text):
        for name in TIER.findall(line):
            tier = CI_TIERS.get(name)
            if tier is None:
                continue
            for command in tier.commands:
                for index, token in enumerate(command):
                    following = command[index + 1] if index + 1 < len(command) else ""
                    if token == "-m" and following and following != "pytest":
                        found.append(following)
    return found


def _expressions(text: str) -> list[str]:
    """Every `-m` expression a pytest invocation in this text selects with."""
    found: list[str] = []
    for line in _logical_lines(text):
        invocations = list(PYTEST.finditer(line))
        for position, invocation in enumerate(invocations):
            end = (
                invocations[position + 1].start()
                if position + 1 < len(invocations)
                else len(line)
            )
            tail = line[invocation.end() : end]
            if match := QUOTED_EXPRESSION.search(tail) or BARE_EXPRESSION.search(tail):
                found.append(match.group(1))
    return found


def _installed_extras(text: str) -> frozenset[str]:
    """The extras this text installs, from an explicit list or a pinned package.

    Only the text after `pip install` is read. A shell block usually starts with
    something like `python -c "import pytest, torch"`, and `pytest` is a real
    distribution in the `dev` extra, so reading the whole line would credit the
    job with an extra it never asked for.
    """
    extras: set[str] = set()
    for line in _logical_lines(text):
        start = line.find("pip install")
        if start < 0:
            continue
        command = line[start:]
        if match := EXTRAS.search(command):
            extras.update(
                extra.strip() for extra in match.group(1).split(",") if extra.strip()
            )
        # A pinned package names the extra carrying it.
        for word in DISTRIBUTION.findall(command):
            extras.update(extras_for(word))
    return frozenset(extras)


def _true_for(expression: str, marks: set[str]) -> bool:
    """Evaluate a pytest marker expression against one test's marks.

    Reading the expression as a set of names would turn `distributed_accel and
    gpu` into "either marker", which hands a device-bound test a lane that has no
    device. Both sides are evaluated eagerly so that short-circuiting cannot
    desynchronize the tokens.
    """
    tokens = re.findall(r"[A-Za-z_]\w*|[()]", expression)
    position = 0

    def peek() -> str:
        return tokens[position] if position < len(tokens) else ""

    def take() -> str:
        nonlocal position
        token = peek()
        position += 1
        return token

    def parse_or() -> bool:
        value = parse_and()
        while peek() == "or":
            take()
            value = parse_and() or value
        return value

    def parse_and() -> bool:
        value = parse_not()
        while peek() == "and":
            take()
            value = parse_not() and value
        return value

    def parse_not() -> bool:
        if peek() == "not":
            take()
            return not parse_not()
        return parse_atom()

    def parse_atom() -> bool:
        token = take()
        if token == "(":
            value = parse_or()
            if peek() == ")":
                take()
            return value
        return token in marks

    return parse_or()


def _selection(text: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """The marker expressions and file paths this text names."""
    expressions = tuple(_expressions(text) + _tier_expressions(text))
    paths = frozenset(
        path
        for line in _logical_lines(text)
        if "pytest" in line
        for path in TEST_FILE.findall(line)
    )
    return expressions, paths


def lanes() -> tuple[Job, ...]:
    return tuple(
        Job(
            workflow=workflow,
            name=name,
            extras=_installed_extras(text),
            expressions=_selection(text)[0],
            paths=_selection(text)[1],
            device=bool(DEVICE_LABEL.search(text)),
        )
        for workflow, name, text in _job_blocks()
    )


def _calls_named(node: ast.AST, names: frozenset[str]) -> set[str]:
    """Import names passed to any call of one of `names` under `node`."""
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not child.args:
            continue
        target = child.func
        called = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", None)
        )
        argument = child.args[0]
        if (
            called in names
            and isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ):
            found.add(argument.value.split(".")[0])
    return found


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


def _is_pytestmark(node: ast.AST) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "pytestmark"
        for target in node.targets
    )


def _module_probes(tree: ast.Module) -> set[str]:
    """What makes the whole module skip, ignoring each test's own decorators."""
    probes: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        probes |= _calls_named(node, IMPORTORSKIP)
        if _is_pytestmark(node):
            probes |= _calls_named(node, SKIP_CONDITION)
    return probes


def _module_marks(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in tree.body:
        if _is_pytestmark(node):
            found |= _marks(node.value)
    return found


def _decorator_marks(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        found |= _marks(decorator)
    return found


def _decorator_probes(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        found |= _calls_named(decorator, SKIP_CONDITION)
    return found


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.name.startswith("test_")
    )


def _tests(tree: ast.Module) -> list[tuple[str, int, set[str], set[str]]]:
    """(name, line, marks, probes) for every test defined in the file."""
    found: list[tuple[str, int, set[str], set[str]]] = []
    for node in tree.body:
        if _is_test(node):
            found.append(
                (
                    node.name,
                    node.lineno,
                    _decorator_marks(node),
                    _calls_named(node, IMPORTORSKIP) | _decorator_probes(node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            class_probes = _decorator_probes(node)
            class_marks = _decorator_marks(node)
            for child in node.body:
                if _is_test(child):
                    found.append(
                        (
                            f"{node.name}::{child.name}",
                            child.lineno,
                            class_marks | _decorator_marks(child),
                            class_probes
                            | _calls_named(child, IMPORTORSKIP)
                            | _decorator_probes(child),
                        )
                    )
    return found


def _audit() -> list[tuple[str, str, int, set[str], set[str]]]:
    """(relative path, test name, line, marks, probes) for every test in the tree."""
    rows: list[tuple[str, str, int, set[str], set[str]]] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        module_marks = _module_marks(tree)
        module_probes = _module_probes(tree)
        for name, line, marks, probes in _tests(tree):
            rows.append(
                (
                    relative,
                    name,
                    line,
                    module_marks | marks,
                    module_probes | probes,
                )
            )
    return rows


def _unmet(probes: set[str], extras: frozenset[str]) -> list[str]:
    """Probed names whose declared extras none of `extras` provides."""
    return sorted(
        name for name in probes if extras_for(name) and not (extras_for(name) & extras)
    )


def test_the_coverage_job_installs_every_integration_it_selects() -> None:
    """A lane that measures must run what it selects, or the numbers are wrong."""
    coverage = next(
        job for job in lanes() if job.workflow == CI_WORKFLOW and job.name == "coverage"
    )

    offenders: list[str] = []
    for path, name, line, marks, probes in _audit():
        if not coverage.selects(marks, path):
            continue
        if unmet := _unmet(probes, coverage.extras):
            offenders.append(f"{path}:{line} {name} (needs {','.join(unmet)})")

    assert not offenders, (
        "the coverage lane selects these tests, but the coverage job does not "
        "install what they need, so they skip and the code they cover is never "
        "measured: " + "; ".join(offenders)
    )


def test_every_integration_a_test_probes_is_installed_by_a_lane_that_runs_it() -> None:
    """A test no lane can run is a test that never runs."""
    jobs = lanes()

    orphans: list[str] = []
    for path, name, line, marks, probes in _audit():
        for integration in sorted(probes):
            providers = extras_for(integration)
            if not providers:
                continue
            runnable = any(
                providers & job.extras and job.runs(marks, path) for job in jobs
            )
            if not runnable:
                orphans.append(
                    f"{path}:{line} {name} probes {integration} "
                    "with no lane that installs it and selects this test"
                )

    assert not orphans, (
        "these tests ask for an optional integration that no lane both installs "
        "and selects, so they skip everywhere and never execute: " + "; ".join(orphans)
    )


def test_the_lane_audit_reads_the_workflows_it_claims_to() -> None:
    """Sanity: the parsers see the real jobs, extras, and expressions."""
    jobs = lanes()
    by_name = {job.name: job for job in jobs}

    assert {"coverage", "triton-optional", "distributed-cpu"} <= set(by_name)

    coverage = by_name["coverage"]
    assert coverage.extras == frozenset({"dev", "jax", "viz", "pennylane"})
    assert coverage.expressions == (
        "(smoke or unit or integration or jax) and not qiskit and not triton",
    )
    # A conjunction stays a conjunction: `and not qiskit` must not read as "any
    # of these names", which is what would hand a skipped test a clean bill.
    assert coverage.selects({"smoke"}, "tests/unrelated.py")
    assert not coverage.selects({"qiskit", "integration"}, "tests/unrelated.py")
    assert not coverage.selects({"triton", "unit"}, "tests/unrelated.py")
    # The coverage lane does select a device-marked test, and cannot run it.
    assert coverage.selects({"gpu", "integration"}, "tests/unrelated.py")
    assert not coverage.runs({"gpu", "integration"}, "tests/unrelated.py")

    triton = by_name["triton-optional"]
    assert "cuda" in triton.extras
    assert triton.expressions == ("triton",)

    # `distributed-cpu` selects through a tier, not an inline expression, and a
    # job that installs Qiskit pins the distribution rather than the extra.
    assert "distributed_cpu" in by_name["distributed-cpu"].expressions
    assert "qiskit" in by_name["qiskit-optional"].extras

    # Reading the whole shell block would credit this job with `dev`, because it
    # starts with `python -c "import pytest, torch"`.
    assert by_name["one-gpu-local"].extras == frozenset()
    assert by_name["one-gpu-local"].device

    # The accelerator job is the only one that can run a device-bound test, and
    # its two commands select different things.
    accelerator = by_name["gpu-scheduled"]
    assert accelerator.device
    assert "cuda" in accelerator.extras
    assert accelerator.expressions == ("distributed_accel and gpu", "triton and gpu")
    assert not accelerator.selects({"gpu"}, "tests/unrelated.py")
    assert accelerator.selects({"gpu", "distributed_accel"}, "tests/unrelated.py")
    assert not by_name["coverage"].device


def test_every_alias_names_a_distribution_the_policy_declares() -> None:
    """An alias that stops resolving would drop its probe out of the audit.

    `extras_for` returns an empty set both for a package no extra declares and
    for an alias pointing at a name that no longer exists, and an empty set means
    "nothing to check". This keeps the second case out of the first's silence.
    """
    unresolved = sorted(
        f"{import_name} -> {distribution}"
        for import_name, distribution in IMPORT_ALIASES.items()
        if distribution_name(distribution) not in PROVIDERS
    )
    assert not unresolved, (
        "these import aliases name a distribution the dependency policy does not "
        "declare, so a test probing them would be skipped by the audit rather "
        "than reported: " + ", ".join(unresolved)
    )

    # And the audits' own integrations resolve, so the aliases above are load
    # bearing rather than decorative.
    for import_name in (
        "braket",
        "jax",
        "jaxlib",
        "pennylane",
        "qiskit",
        "qiskit_aer",
        "triton",
    ):
        assert extras_for(import_name), import_name
