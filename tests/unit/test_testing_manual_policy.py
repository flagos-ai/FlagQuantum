import re
from pathlib import Path

import pytest

from tools.check_dependency_policy import load_toml
from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit

TESTING_MANUAL = "docs/development/TESTING.md"
CI_WORKFLOW = ".github/workflows/ci.yml"
COVERAGE_POLICY = "contracts/coverage-policy.toml"

# Spelled-out numbers, for the counts the manual states in prose.
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_testing_manual_has_human_entrypoints():
    text = _read("docs/development/TESTING.md")
    for heading in (
        "## Quick Start",
        "## Marker Reference",
        "## Test Tiers",
        "## CI Policy",
        "## Path Classification Mapping",
        "## Non-Negotiable Release Boundary",
    ):
        assert heading in text


def test_testing_manual_distinguishes_required_workflows():
    text = _read("docs/development/TESTING.md")
    for phrase in (
        "Daily Development",
        "Local API And Runtime Work",
        "Distributed CPU Semantics",
        "Real Multi-GPU Or Accelerator Work",
        "Multi-Node Work",
        "Benchmark Release Validation",
    ):
        assert phrase in text


def test_testing_manual_lists_current_commands():
    text = _read("docs/development/TESTING.md")
    for command in (
        "python tools/ci_tier.py pr-default",
        "python tools/ci_tier.py pr-runtime",
        "python tools/ci_tier.py pr-distributed",
        "python tools/ci_tier.py gpu-scheduled",
        "python tools/ci_tier.py multinode-scheduled",
        "python tools/ci_tier.py release",
        'python -m pytest -m "smoke or unit" -q',
        'python -m pytest -m "integration" -q',
        'python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q',
        'python -m pytest -m "distributed_accel and gpu" -q',
        'python -m pytest -m "distributed_multinode" -q',
        "python benchmarks/audit_results.py --input benchmarks/results",
        "python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability",
    ):
        assert command in text


def _registered_markers() -> list[str]:
    """Read the marker names declared in pytest.ini's ``markers`` block."""
    declarers = ("    ", "\t")
    markers: list[str] = []
    in_block = False
    for line in _read("pytest.ini").splitlines():
        if line.strip() == "markers =":
            in_block = True
            continue
        if not in_block:
            continue
        if not line.startswith(declarers) or ":" not in line:
            if line.strip():
                break
            continue
        markers.append(line.strip().split(":", 1)[0].strip())
    return markers


def test_testing_manual_documents_every_marker_with_proof_boundary():
    text = _read("docs/development/TESTING.md")
    registered = _registered_markers()

    assert "jax" in registered, "the guard should read the marker block"
    for marker in registered:
        assert (
            f"| `{marker}` |" in text
        ), f"TESTING.md does not document marker {marker}"
    assert "| Marker | Runtime Environment | Proves | Does Not Prove |" in text
    assert "Real benchmark performance or hardware behavior" in text
    assert "Release readiness or scalability on its own" in text


def test_testing_manual_matches_ci_tier_commands_and_boundaries():
    text = _read("docs/development/TESTING.md")
    normalized = " ".join(text.split())

    for tier_name, tier in CI_TIERS.items():
        assert f"python tools/ci_tier.py {tier_name}" in text
        for command_line in tier.command_lines():
            if tier_name in {"nightly", "release"} and command_line.startswith(
                "python -m pytest"
            ):
                continue
            if command_line.startswith("python -m pytest"):
                quoted_marker = command_line.replace(
                    "python -m pytest -m ", 'python -m pytest -m "'
                ).replace(" -q", '" -q')
                assert command_line in normalized or quoted_marker in text, (
                    tier_name,
                    command_line,
                )
            else:
                assert command_line in text, (tier_name, command_line)
        assert (
            tier.does_not_prove.split(".")[0] in normalized
            or "not scalability evidence" in normalized
        )


def test_testing_manual_preserves_scalability_boundary():
    text = " ".join(_read("docs/development/TESTING.md").split())
    assert "CPU distributed tests prove semantics and fail-closed behavior only" in text
    assert "They do not prove real multi-card capacity expansion" in text
    assert (
        "Never use CPU distributed tests alone as scalability release evidence" in text
    )
    assert (
        "Release-grade scalability requires one logical workload sharded across ranks"
        in text
    )
    assert 'distribution_semantics="sharded_across_ranks"' in text
    assert "sharded gradient/optimizer ownership" in text


def test_testing_manual_aligned_with_agents():
    docs = _read("docs/development/TESTING.md")
    agents = _read("AGENTS.md")
    normalized_docs = " ".join(docs.split())
    assert "python tools/ci_tier.py pr-default" in agents
    assert "python tools/ci_tier.py pr-default" in docs
    assert (
        "Never use CPU distributed tests alone as scalability release evidence"
        in agents
    )
    assert (
        "Never use CPU distributed tests alone as scalability release evidence"
        in normalized_docs
    )


def _ci_jobs() -> list[str]:
    """The job names declared in `.github/workflows/ci.yml`."""
    jobs: list[str] = []
    in_jobs = False
    for line in _read(CI_WORKFLOW).splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if match:
            jobs.append(match.group(1))
    return jobs


def test_testing_manual_names_only_test_files_that_exist() -> None:
    """A path in the manual is an instruction, so a dangling one sends a reader nowhere.

    This is how `tests/unit/test_marker_seeding_policy.py` was found: the per-file
    marker guard was replaced by a per-test one, the manual kept naming the old
    file, and nothing in the repository noticed.
    """
    text = _read(TESTING_MANUAL)
    missing = [
        path
        for path in sorted(set(re.findall(r"\btests/[\w./-]+\.py\b", text)))
        if not Path(path).is_file()
    ]

    assert (
        not missing
    ), "the manual names these test paths, but they do not exist: " + ", ".join(missing)


def test_testing_manual_names_every_job_ci_defines() -> None:
    """A job nobody documented is a job whose coverage of the policy is unstated.

    `triton-optional`, `dependency-bounds`, `coverage`, and `supply-chain` all ran
    in `ci.yml` while the manual described a list that did not include them.

    The name has to appear as a code span. A bare-word search passes on prose
    that happens to share the name -- "supply-chain checks" appears in the push
    gate paragraph -- so deleting the job's own entry left it green.
    """
    text = _read(TESTING_MANUAL)
    jobs = _ci_jobs()
    assert {"quality", "coverage"} <= set(jobs), "the guard should read ci.yml"

    undocumented = [job for job in jobs if f"`{job}`" not in text]

    assert (
        not undocumented
    ), "ci.yml defines these jobs and the manual never names them: " + ", ".join(
        undocumented
    )

    # The sentence states a count as well as a list, and the count is the other
    # half of what went stale here: it read "five primary CPU jobs" while the
    # list under it named six and the file defined eleven.
    stated = re.search(r"`ci\.yml` defines (\w+) jobs", text)
    assert stated, "the manual should say how many jobs ci.yml defines"
    assert stated.group(1) in NUMBER_WORDS, f"unrecognized number {stated.group(1)!r}"
    assert NUMBER_WORDS[stated.group(1)] == len(jobs), (
        f"the manual says ci.yml defines {stated.group(1)} jobs, but it defines "
        f"{len(jobs)}: {', '.join(jobs)}"
    )


def test_testing_manual_states_the_coverage_floor_ci_enforces() -> None:
    """The manual's one enforceable number has to be the number being enforced.

    It read "55%, below the measured 56% baseline" while the policy enforced 76,
    because four stages raised the floor and none of them came back here.
    """
    enforced = load_toml(Path(COVERAGE_POLICY))["global"]["min"]
    stated = [
        int(value) for value in re.findall(r"floor is (\d+)%", _read(TESTING_MANUAL))
    ]

    assert stated, "the manual should state the global coverage floor"
    assert stated == [enforced], (
        f"the manual states the global coverage floor as {stated}, but "
        f"{COVERAGE_POLICY} enforces {enforced}%"
    )
