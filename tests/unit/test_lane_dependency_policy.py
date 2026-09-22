"""Lane dependency policy: can any lane actually run the test it selects?

The workflows select tests by marker and install dependencies separately. A test
that calls `pytest.importorskip` records a skip when its package is missing, and
pytest reports a skip the same way it reports a pass. So a lane can select a test
it cannot run and still come up green. Four failures of that shape were found
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
- Six more tests in `tests/unit/test_real_imag_kernels.py` were guarded by
  `skipif(not torch.cuda.is_available())` without a `gpu` mark. The CPU lanes
  selected and skipped them; the accelerator lane selects `gpu`, so it never
  saw them. They now carry `gpu` and `triton`.
- `tests/test_tensor_network.py` probed `cotengra`, and no extra declared it. The
  external planning backend under test is a real optional integration, so the
  answer was to declare it rather than to drop the test: `cotengra` is an extra
  and the coverage job installs it.

Five invariants. The first is narrower and stronger: a lane whose job is to
measure must run what it selects, or its measurement is a lie. The second: every
optional integration a test asks for must be installed by *some* lane that
selects that test. The third: a test that waits on a device must be within reach
of a lane that has one, because no install line can supply it. The fourth closes
the hole the second leaves: a probe for a package the policy mentions nowhere is
not "no lane installs it" but "no lane can be wired to install it", so the answer
has to be recorded in the policy instead of implied by silence. The fifth applies
that reasoning to the other thing no lane can supply: a test that skips unless an
environment variable is set is opt-in by design, and the variable is the only way
in, so the testing manual's opt-in table is where it has to be named.

Scope: the first three reason about what a lane can be configured to provide,
and the fourth and fifth about what no lane can provide at all. A value only a
person can supply is not a gap in the wiring; leaving it unrecorded is. The
fourth is about the policy rather than the lanes, which is what made the
`cotengra` probe above invisible to the second.

What a lane runs is read from the `-m` expressions in the workflows and from the
`tools/ci_tier.py` tiers they invoke, so a job whose selection lives in a tier is
seen too.
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
# Where an opt-in device run is recorded, since no lane can supply its variable.
MANUAL = ROOT / "docs/development/TESTING.md"

# The policy names distributions, the tests probe import names, and the two
# usually agree once both are normalized. These are the places they do not: a
# probe for the import name has to resolve to the distribution that provides it,
# or the test would go unchecked rather than reported. The sanity test below
# fails if any entry stops naming a declared distribution, so a rename shows up
# as a failure instead of as silence.
IMPORT_ALIASES = {
    "braket": "amazon-braket-sdk",
    "cirq": "cirq-core",
    "cudaq": "cudaq",
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


# The project's own top-level package. A test that probes it is asking whether an
# optional module of this repository imports, which is not a dependency question.
INTERNAL = "flagquantum"


def _placed(name: str) -> bool:
    """Whether the policy places this probe: core, an extra, or the project itself.

    `extras_for` cannot answer this: it returns an empty set both for a package
    nothing declares and for a core distribution, and the two want opposite
    verdicts. A probe is read at its top level, so `torch.distributed` is core
    and `flagquantum.core.circuit` is the project's own module rather than a
    distribution called `flagquantum-core-circuit`.
    """
    root = name.split(".")[0]
    if root == INTERNAL:
        return True
    normalized = distribution_name(IMPORT_ALIASES.get(root, root))
    return normalized in CORE or normalized in PROVIDERS


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


def _guards_on_cuda(decorator: ast.AST) -> bool:
    """Whether this decorator's argument asks about a CUDA device.

    The test is the attribute chain, not a bare name: in
    `torch.cuda.is_available()` the word `cuda` is an attribute of an attribute,
    so searching for a Name called `cuda` finds nothing.
    """
    return any(
        isinstance(child, ast.Attribute)
        and child.attr in {"is_available", "device_count"}
        and isinstance(child.value, ast.Attribute)
        and child.value.attr == "cuda"
        for child in ast.walk(decorator)
    )


def _requires_device(node: ast.AST) -> bool:
    """Whether this test waits on a CUDA device before it can run.

    Two spellings count. A `skipif` decorator on a CUDA check is the declarative
    one. The other is a `pytest.skip` in the body next to a CUDA check, which is
    the same statement written as control flow -- it is read coarsely, as "this
    test mentions the device and can skip", because telling the two apart would
    mean evaluating the guard.

    A test that calls a helper which itself skips on the device is a third shape
    and this misses it; `tests/test_statevector_triton_gates.py` is the one
    instance, and it carries the marks regardless.
    """
    if any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "skipif"
        and _guards_on_cuda(decorator)
        for decorator in getattr(node, "decorator_list", [])
    ):
        return True

    skips = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "skip"
        for child in ast.walk(node)
    )
    return skips and _guards_on_cuda(node)


def _calls_skip(node: ast.AST) -> bool:
    """Whether this node calls `pytest.skip`."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "skip"
        for child in ast.walk(node)
    )


def _environment_reads(node: ast.AST) -> set[str]:
    """The environment variable names this node reads.

    Three spellings, all of which appear in this tree: `os.environ.get("X")`,
    `os.environ["X"]`, and `os.getenv("X")`. The last one is what an earlier
    grep for `environ` missed.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Attribute) and target.attr in {"get", "getenv"}:
                base = target.value
                reads_environ = (
                    isinstance(base, ast.Attribute) and base.attr == "environ"
                ) or (isinstance(base, ast.Name) and base.id in {"os", "environ"})
                if reads_environ and child.args:
                    key = child.args[0]
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        found.add(key.value)
        elif isinstance(child, ast.Subscript):
            base = child.value
            if isinstance(base, ast.Attribute) and base.attr == "environ":
                key = child.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found.add(key.value)
    return found


def _environment_condition(node: ast.AST) -> frozenset[str]:
    """Environment variables a test must be given before it will run.

    A read is a gate only when the same test can skip. `FQ_TEST_CHECKPOINT` and
    `LOCAL_RANK` are read by tests that always run, and those are configuration
    rather than opt-in -- reporting them would bury the real ones.
    """
    if not _calls_skip(node):
        return frozenset()
    return frozenset(_environment_reads(node))


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.name.startswith("test_")
    )


def _tests(
    tree: ast.Module,
) -> list[tuple[str, int, set[str], set[str], bool, frozenset[str]]]:
    """(name, line, marks, probes, device, environment) for every test in the file."""
    found: list[tuple[str, int, set[str], set[str], bool, frozenset[str]]] = []
    for node in tree.body:
        if _is_test(node):
            found.append(
                (
                    node.name,
                    node.lineno,
                    _decorator_marks(node),
                    _calls_named(node, IMPORTORSKIP) | _decorator_probes(node),
                    _requires_device(node),
                    _environment_condition(node),
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
                            _requires_device(child),
                            _environment_condition(child),
                        )
                    )
    return found


class Row(NamedTuple):
    """One test, and everything the invariants below need to judge it."""

    path: str
    name: str
    line: int
    marks: set[str]
    probes: set[str]
    requires_device: bool
    environment: frozenset[str]


def _audit() -> list[Row]:
    rows: list[Row] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        module_marks = _module_marks(tree)
        module_probes = _module_probes(tree)
        for name, line, marks, probes, requires_device, environment in _tests(tree):
            rows.append(
                Row(
                    path=relative,
                    name=name,
                    line=line,
                    marks=module_marks | marks,
                    probes=module_probes | probes,
                    requires_device=requires_device,
                    environment=environment,
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
    for row in _audit():
        if not coverage.selects(row.marks, row.path):
            continue
        if unmet := _unmet(row.probes, coverage.extras):
            offenders.append(
                f"{row.path}:{row.line} {row.name} (needs {','.join(unmet)})"
            )

    assert not offenders, (
        "the coverage lane selects these tests, but the coverage job does not "
        "install what they need, so they skip and the code they cover is never "
        "measured: " + "; ".join(offenders)
    )


def test_every_integration_a_test_probes_is_installed_by_a_lane_that_runs_it() -> None:
    """A test no lane can run is a test that never runs."""
    jobs = lanes()

    orphans: list[str] = []
    for row in _audit():
        for integration in sorted(row.probes):
            providers = extras_for(integration)
            if not providers:
                continue
            runnable = any(
                providers & job.extras and job.runs(row.marks, row.path) for job in jobs
            )
            if not runnable:
                orphans.append(
                    f"{row.path}:{row.line} {row.name} probes {integration} "
                    "with no lane that installs it and selects this test"
                )

    assert not orphans, (
        "these tests ask for an optional integration that no lane both installs "
        "and selects, so they skip everywhere and never execute: " + "; ".join(orphans)
    )


def test_every_package_a_test_probes_is_declared_by_the_policy() -> None:
    """The gap invariant two cannot see: a package nothing declares.

    Invariant two asks whether some lane installs what a test asks for, but it
    reads `extras_for`, which is empty when the policy mentions the package
    nowhere at all -- so it skips exactly the probes it cannot resolve. The
    answer to "no lane installs this" has to be a decision recorded in the
    policy: declare the extra, or drop the probe. Left unsaid it is a skip in
    every lane there is, which reads the same as a pass.
    """
    offenders: list[str] = []
    for row in _audit():
        for probe in sorted(row.probes):
            if _placed(probe):
                continue
            offenders.append(f"{row.path}:{row.line} {row.name} probes {probe}")

    assert not offenders, (
        "these tests ask for a package no extra and no core dependency declares, "
        "so they skip in every lane and no lane can be wired to fix it: "
        + "; ".join(offenders)
    )


def test_the_declared_probe_predicate_separates_omission_from_core() -> None:
    """`_placed` has to give the two empty-`extras_for` cases opposite answers.

    The predicate is the whole check, and it reads like `extras_for`, so a
    change that collapses the two would leave the invariant above passing while
    it stopped looking at anything. The undeclared name is deliberately one no
    policy will ever place: `cotengra` itself was the real instance, and pinning
    the behaviour to it would make this test pass by being fixed rather than by
    staying correct.
    """
    assert _placed("a-package-no-extra-declares") is False
    assert _placed("matplotlib") is True  # declared by `viz`
    assert _placed("cotengra") is True  # declared by `cotengra`
    assert _placed("torch") is True  # core
    assert _placed("torch.distributed") is True  # core, read at its top level
    assert _placed("flagquantum") is True  # the project itself
    assert _placed("flagquantum.core.circuit") is True  # a module of the project
    assert _placed("qiskit_aer") is True  # normalized to `qiskit-aer`


def test_a_test_that_waits_on_a_device_is_within_reach_of_one() -> None:
    """Hardware is the third axis a lane can fail to provide.

    A test that skips on `not torch.cuda.is_available()` cannot be rescued by any
    install line. It needs a lane whose runner has the device, and the only lane
    like that selects `gpu`. Six tests in `tests/unit/test_real_imag_kernels.py`
    carried the guard without the marker, so every CPU lane selected them and
    skipped them while the accelerator lane never selected them at all.
    """
    jobs = lanes()
    accelerators = [job for job in jobs if job.device]

    assert accelerators, "no job in any workflow runs on a device runner"

    orphans: list[str] = []
    for row in _audit():
        if not row.requires_device:
            continue
        if any(job.runs(row.marks, row.path) for job in accelerators):
            continue
        orphans.append(
            f"{row.path}:{row.line} {row.name} (marks={','.join(sorted(row.marks))})"
        )

    assert not orphans, (
        "these tests skip on a device check, but no lane that has a device "
        "selects them, so they skip in every lane there is: " + "; ".join(orphans)
    )


def test_a_test_that_waits_on_an_environment_variable_is_named_in_the_manual() -> None:
    """No lane can set the variable a device run needs, so the manual must name it.

    The same reasoning as the two invariants above, applied to the one input a
    lane cannot supply at all. A test that skips unless `VAR` is set is skipped
    in every lane by construction, which reads exactly like a pass, so the only
    place its existence survives is a document someone reads. Nine of the
    thirteen in this tree were named nowhere when this check was written; four
    were named only in a reference page that describes the tool rather than the
    test.
    """
    manual = MANUAL.read_text(encoding="utf-8")

    orphans: list[str] = []
    for row in _audit():
        for variable in sorted(row.environment):
            if variable not in manual:
                orphans.append(f"{row.path}:{row.line} {row.name} (needs {variable})")

    assert not orphans, (
        "these tests skip unless an environment variable is set, and no lane sets "
        "one, so a reader who is not told the variable cannot run them at all: "
        + "; ".join(orphans)
    )


def test_the_environment_condition_separates_a_gate_from_configuration() -> None:
    """A read only counts when the same test can skip, and all three spellings count.

    Without the first half this reports `FQ_TEST_CHECKPOINT` and `LOCAL_RANK`,
    which tests read to configure themselves and never gate on, and the real
    entries drown. Without the second it misses the one test that calls
    `os.getenv` instead of `os.environ.get`.
    """
    source = (
        "import os\n"
        "import pytest\n"
        "def test_gate():\n"
        "    if os.environ.get('A') is None:\n"
        "        pytest.skip('set A')\n"
        "def test_gate_by_subscript():\n"
        "    if not os.environ['B']:\n"
        "        pytest.skip('set B')\n"
        "def test_gate_by_getenv():\n"
        "    if os.getenv('C') is None:\n"
        "        pytest.skip('set C')\n"
        "def test_configuration_only():\n"
        "    root = os.environ['D']\n"
        "    assert root\n"
        "def test_reading_without_skipping():\n"
        "    assert os.environ.get('E', 'default')\n"
    )
    tree = ast.parse(source)
    found = {name: environment for name, _, _, _, _, environment in _tests(tree)}

    assert found["test_gate"] == frozenset({"A"})
    assert found["test_gate_by_subscript"] == frozenset({"B"})
    assert found["test_gate_by_getenv"] == frozenset({"C"})
    assert found["test_configuration_only"] == frozenset()
    assert found["test_reading_without_skipping"] == frozenset()


def test_the_lane_audit_reads_the_workflows_it_claims_to() -> None:
    """Sanity: the parsers see the real jobs, extras, and expressions."""
    jobs = lanes()
    by_name = {job.name: job for job in jobs}

    assert {"coverage", "triton-optional", "distributed-cpu"} <= set(by_name)

    coverage = by_name["coverage"]
    assert coverage.extras == frozenset(
        {"dev", "jax", "viz", "cirq", "pennylane", "cotengra"}
    )
    assert coverage.expressions == (
        "(smoke or unit or integration or jax) and not qiskit and not cudaq and not triton",
    )
    # A conjunction stays a conjunction: `and not qiskit` must not read as "any
    # of these names", which is what would hand a skipped test a clean bill.
    assert coverage.selects({"smoke"}, "tests/unrelated.py")
    assert not coverage.selects({"qiskit", "integration"}, "tests/unrelated.py")
    assert not coverage.selects({"cudaq", "integration"}, "tests/unrelated.py")
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
    assert not coverage.device


def test_the_device_guard_detector_recognizes_the_guard() -> None:
    """The device invariant is only as good as this predicate.

    It was written once with a search for a `Name` called `cuda`, which finds
    nothing: in `torch.cuda.is_available()` the word is an attribute of an
    attribute. The predicate returned False for every test, and the invariant
    above passed with the marks removed from the six tests it exists to protect.
    Both known spellings are pinned here so that cannot recur silently.
    """
    source = "\n".join(
        (
            "import pytest, torch",
            "",
            "@pytest.mark.skipif(not torch.cuda.is_available(), reason='x')",
            "def test_availability(): pass",
            "",
            "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='x')",
            "def test_device_count(): pass",
            "",
            "def test_body_form():",
            "    if not torch.cuda.is_available():",
            "        pytest.skip('cuda')",
            "",
            "@pytest.mark.skipif(sys.platform == 'win32', reason='x')",
            "def test_platform(): pass",
            "",
            "def test_mentions_cuda_without_skipping():",
            "    return torch.cuda.is_available()",
            "",
        )
    )
    tree = ast.parse(source)
    tests = {node.name: node for node in tree.body if _is_test(node)}

    assert _requires_device(tests["test_availability"])
    assert _requires_device(tests["test_device_count"])
    assert _requires_device(tests["test_body_form"])
    assert not _requires_device(tests["test_platform"])
    assert not _requires_device(tests["test_mentions_cuda_without_skipping"])


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
