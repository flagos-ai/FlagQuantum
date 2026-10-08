#!/usr/bin/env python3
"""Validate the checked-in branch-protection required-check contract.

`.github/required-checks.json` records the checks branch protection on `main`
requires. Two ways for that record to be wrong both matter here, and both have
happened in this repository:

- a required check that no workflow job produces blocks every merge, because it
  stays `Expected` forever;
- a required job that the workflows renamed or dropped stops gating silently,
  because a check nobody requires is a check nobody reads.

So the contract is read against the workflow documents rather than against a
second copy of the job names. Every required name must be a check some job in
`.github/workflows/` produces, and the names GitHub derives for a matrix job are
resolved from the matrix itself.

The name is the **check run's** name, which is the job's key -- or the job's
`name:` when it declares one -- with GitHub's matrix suffix appended, and **not**
`"<workflow> / <job>"`. That distinction was measured rather than assumed: branch
protection was pointed at the `"<workflow> / <job>"` form of these eight names on
2026-10-08 and pull request #597 stayed `blocked` with all nineteen of its check
runs `success` and no other requirement unmet, because no check run is ever
reported under that name. The same eight contexts written as the job names alone
reported as required and the pull request went `clean`.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

SCHEMA = "flagquantum_required_checks_v1"

# The checks required on `main`, named as GitHub reports them. A job that merely
# exists is not gated; a job that is gated must exist. Both halves are asserted
# below.
REQUIRED_CHECKS = frozenset(
    {
        "quality",
        "cpu-core (3.10)",
        "cpu-core (3.11)",
        "cpu-core (3.12)",
        "package",
        "distributed-cpu",
        "coverage",
        "pre-commit",
    }
)


def _matrix_suffixes(matrix: object) -> list[str]:
    """The parenthesised suffixes GitHub appends to a matrix job's name.

    GitHub joins the values of every matrix key with `, ` in declaration order,
    so `cpu-core` with one interpreter becomes `cpu-core (3.10)` and a job whose
    matrix is written as `include:` entries is named after the values of each
    entry. A non-matrix job produces the bare job name.
    """

    if not isinstance(matrix, dict):
        return [""]
    include = matrix.get("include")
    if isinstance(include, list):
        return [
            f" ({', '.join(str(value) for value in entry.values())})"
            for entry in include
            if isinstance(entry, dict)
        ]
    keys = [key for key, values in matrix.items() if isinstance(values, list)]
    if not keys:
        return [""]
    suffixes = [""]
    for key in keys:
        suffixes = [
            f"{prefix}{', ' if prefix else ''}{value}"
            for prefix in suffixes
            for value in matrix[key]
        ]
    return [f" ({suffix})" for suffix in suffixes]


def _check_name(path: Path, job: str, body: dict) -> str:
    """The name GitHub reports a job's check run under.

    A job without `name:` is reported under its key. A job with a literal `name:`
    is reported under that instead, and a `name:` holding an expression is
    refused: this file resolves names from the documents, so a name it cannot
    resolve is a required check that would be recorded unchecked.
    """

    declared = body.get("name")
    if declared is None:
        return job
    if not isinstance(declared, str) or "${{" in declared:
        raise SystemExit(
            f"{path.name}:{job}: the job name is interpolated, so this gate cannot "
            "resolve the check name and will not guess it"
        )
    return declared


def _produced_checks(root: Path) -> set[str]:
    """Every check name the workflows in this repository can report."""

    produced: set[str] = set()
    for path in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job, body in jobs.items():
            if not isinstance(body, dict):
                raise SystemExit(f"{path.name}:{job}: not a job mapping")
            name = _check_name(path, job, body)
            strategy = body.get("strategy")
            matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
            for suffix in _matrix_suffixes(matrix):
                produced.add(f"{name}{suffix}")
    return produced


def declared_checks(root: Path) -> tuple[str, ...]:
    """The required-check names recorded for `root`, or a refusal.

    The record is validated against the policy constant here rather than only in
    `main`, so a caller that reads the contract gets the same refusal the gate
    does instead of a list that quietly disagrees with the policy.
    """

    payload = json.loads((root / ".github/required-checks.json").read_text())
    checks = tuple(payload.get("checks", ()))
    missing = REQUIRED_CHECKS - set(checks)
    unexpected = set(checks) - REQUIRED_CHECKS
    if (
        payload.get("schema") != SCHEMA
        or missing
        or unexpected
        or len(checks) != len(set(checks))
    ):
        raise SystemExit(
            "invalid required-check policy; "
            f"missing={sorted(missing)}; unexpected={sorted(unexpected)}"
        )
    return checks


def unproduced_checks(root: Path) -> list[str]:
    """Required checks that no workflow job in `root` reports.

    These are the names that would sit `Expected` forever under branch
    protection, blocking every merge, which is why the gate refuses them.
    """

    return sorted(set(declared_checks(root)) - _produced_checks(root))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks = declared_checks(root)
    unproduced = unproduced_checks(root)
    if unproduced:
        raise SystemExit(
            "required check names no workflow job produces, so they would stay "
            f"Expected forever: {unproduced}"
        )

    gpu = (root / ".github/workflows/local-gpu.yml").read_text()
    if "two-gpu-distributed-required:" not in gpu:
        raise SystemExit("required two-GPU workflow job missing")
    trigger = gpu.split("jobs:", 1)[0]
    if "workflow_dispatch:" not in trigger or "pull_request" in trigger:
        raise SystemExit("GPU workflow must be manually dispatched")
    print(
        f"validated {len(checks)} externally configured required checks "
        f"against {len(_produced_checks(root))} checks the workflows produce"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
