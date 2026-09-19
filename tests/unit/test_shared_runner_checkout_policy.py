"""A self-hosted runner's clone outlives the job that used it.

`actions/checkout@v4` cannot undo a sparse checkout on the git these runners
carry (2.34). It does call `git sparse-checkout disable`, and that does restore
the files and clear the skip-worktree bits -- but it leaves `core.sparseCheckout`
set to `true` and the pattern file in `.git/info/sparse-checkout` in place, so
the `git checkout --force -B <branch> <sha>` that follows in the same step
re-applies those patterns and prunes the tree again. `git clean -ffdx` and
`git reset --hard` do not repair it either: both honour the skip-worktree bits.
The checkout reports success the whole way.

So a job that asks for a sparse checkout decides what the *next* job on that
host can see. `local-gpu.yml` asks for `flagquantum`, `tests`, `tools`, one
benchmark, and the packaging files; the scheduled lane runs on the same two
hosts and needs two benchmarks that list does not name. Its first six phases
passed -- `tests/` was in the list -- and the seventh died with `[Errno 2] No
such file or directory` from a `python` that had been handed a path the
checkout had removed. The lane has never got past that phase, and the phase
after it (`benchmarks/continuous_performance.py`) would have run, because that
one file is named.

A job therefore has to be deterministic about its clone in one of two ways: say
what it wants with `sparse-checkout:`, or undo whatever was already there. This
module requires the second of every job that does not do the first, and holds
the three commands to their measured order and content.

The step cannot be shared. It has to run before the checkout, so it cannot be a
script in `tools/` -- `tools/` is not on the runner's disk until the checkout it
would have to precede -- and a composite action in this repository has the same
problem. It is therefore written out in each job, and pinned here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# What is left in the clone is a *pair*: the config key and the pattern file.
# `git sparse-checkout disable` alone leaves both, and removing the file alone
# leaves `core.sparseCheckout` set, so a copy that drops either one is a copy
# that does not work -- measured on the runner's git, by replaying the checkout
# sequence: with `disable` omitted the pattern file re-applies on the next
# `git checkout`; with the `rm` omitted the file is still there to be read.
CLEARING_COMMANDS = (
    "git sparse-checkout disable",
    "rm -f .git/info/sparse-checkout",
    "git config --local --unset core.sparseCheckout",
)

# The guard matters: the workspace is empty on a runner that has never run this
# repository, and a bare `git sparse-checkout disable` would then fail the step.
GUARD = "-d .git"


def _jobs(path: Path) -> dict[str, dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = document.get("jobs")
    assert isinstance(jobs, dict), f"{path.name}: no jobs mapping"
    return jobs


def _labels(job: dict[str, object]) -> tuple[str, ...]:
    value = job.get("runs-on")
    if isinstance(value, str):
        return (value,)
    assert isinstance(value, list), f"unexpected runs-on: {value!r}"
    return tuple(str(item) for item in value)


def _self_hosted_jobs() -> dict[str, tuple[Path, dict[str, object]]]:
    """Every job that checks a repository out onto a runner we own."""

    found: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in _jobs(path).items():
            if "self-hosted" in {label.lower() for label in _labels(job)}:
                found[f"{path.name}:{name}"] = (path, job)
    return found


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list), f"unexpected steps: {steps!r}"
    return [step for step in steps if isinstance(step, dict)]


def _is_checkout(step: dict[str, object]) -> bool:
    return str(step.get("uses", "")).startswith("actions/checkout")


def _declares_sparse_checkout(step: dict[str, object]) -> bool:
    settings = step.get("with")
    return isinstance(settings, dict) and bool(settings.get("sparse-checkout"))


def _clears_the_leftover(step: dict[str, object]) -> bool:
    run = step.get("run")
    return (
        isinstance(run, str)
        and GUARD in run
        and all(command in run for command in CLEARING_COMMANDS)
    )


def _offence(job: dict[str, object]) -> str | None:
    """What this job gets wrong about the clone it asks for, if anything."""

    steps = _steps(job)
    checkouts = [index for index, step in enumerate(steps) if _is_checkout(step)]
    if not checkouts:
        return None
    # A job that names its own paths already decides what it sees: whatever the
    # last job left is overwritten by `git sparse-checkout set` before the tree
    # is read. Asking it to clear first would be asking it to do nothing.
    if any(_declares_sparse_checkout(steps[index]) for index in checkouts):
        return None
    if any(_clears_the_leftover(step) for step in steps[: checkouts[0]]):
        return None
    return (
        "checks the repository out onto a self-hosted runner without declaring "
        "`sparse-checkout:` and without clearing the sparse state a previous "
        "job on that host may have left"
    )


def test_the_scan_finds_the_self_hosted_jobs_it_is_meant_to_police() -> None:
    # A scan that found nothing would satisfy the rule below while checking
    # nothing, which is the same failure as the one being guarded against.
    found = set(_self_hosted_jobs())

    assert {
        "flagos-reference.yml:statevector-local-p0",
        "local-gpu.yml:one-gpu-local",
        "local-gpu.yml:two-gpu-distributed-required",
        "scheduled-hardware.yml:crossover-scheduled",
        "scheduled-hardware.yml:gpu-scheduled",
        "scheduled-hardware.yml:local-scale-scheduled",
        "scheduled-hardware.yml:multinode",
    } <= found


def test_every_undeclared_checkout_clears_the_state_a_previous_job_left() -> None:
    offenders: list[str] = []
    for key, (_path, job) in sorted(_self_hosted_jobs().items()):
        offence = _offence(job)
        if offence is not None:
            offenders.append(f"{key} {offence}")

    assert not offenders, (
        "the clone belongs to the runner, not to the job, so a job that does "
        "not name its own paths has to undo whatever the last one left:\n"
        + "\n".join(offenders)
    )


def test_a_job_that_names_its_own_paths_is_not_asked_to_clear_them() -> None:
    # The control on the other side: the rule is not "every job clears the
    # clone". A job whose checkout names its paths is already deterministic
    # against whatever was there, which is why the two `local-gpu.yml` jobs and
    # the multi-node job -- the ones that write a pattern list -- need no step.
    _path, declared = _self_hosted_jobs()["scheduled-hardware.yml:multinode"]

    assert _offence(declared) is None


def test_a_job_that_clears_only_part_of_the_state_is_reported() -> None:
    # Each of the three commands is load-bearing, so a copy that keeps one of
    # them and drops another has to be reported rather than counted as cleared.
    # This is the check that the rule can fail, since every job in the tree
    # currently passes it.
    def job(run: str) -> dict[str, object]:
        return {
            "steps": [
                {"name": "Clear", "run": run},
                {"uses": "actions/checkout@v4"},
            ]
        }

    everything = "\n".join(["if [ -d .git ]; then", *CLEARING_COMMANDS, "fi"])

    assert _offence(job(everything)) is None
    for dropped in CLEARING_COMMANDS:
        kept = "\n".join(command for command in CLEARING_COMMANDS if command != dropped)
        assert _offence(job(kept)) is not None, dropped
    assert _offence(job(everything.replace(GUARD, "-e .git"))) is not None


def test_a_clear_step_after_the_checkout_is_reported() -> None:
    # The step has to come first: it exists so that the checkout below it is the
    # plain, complete checkout it says it is, and a `run:` step placed after it
    # repairs the tree too late for anything the checkout itself did with it.
    steps = [
        {"uses": "actions/checkout@v4"},
        {
            "run": "\n".join(["if [ -d .git ]; then", *CLEARING_COMMANDS, "fi"]),
        },
    ]

    assert _offence({"steps": steps}) is not None
