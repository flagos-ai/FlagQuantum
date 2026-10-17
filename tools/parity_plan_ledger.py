#!/usr/bin/env python3
"""Validate the checked-in row ledger of the CUDA-Q parity plan.

The plan itself is prose and it is not in this repository. Its blocking table
names the accelerator rows with an ellipsis -- ``W2-01``…``W2-08`` -- so the
prose cannot be read into an exact set of unfinished rows by any parser, and a
grep for a landing marker cannot tell a row that is blocked from a row that is
merely unannotated. Both of those facts have produced wrong counts in the plan
itself. This ledger is the machine-readable record that replaces reading the
prose: one entry per row, enumerated rather than abbreviated.

What the ledger deliberately does not carry:

- **A team column.** Path ownership already has one authority,
  ``team-ownership.toml``, and a second copy of it here would drift from it.
  Several plan rows name a team the roster does not declare; recording that
  copy here would preserve the disagreement instead of surfacing it at the
  place that owns it.
- **Titles, narratives, and commit identifiers.** Those are prose. The ledger
  answers a different question: which rows exist, what each delivers, what it
  waits for, which gate proves it, and whether it is done.

The one field that exists for a bookkeeping gap rather than for a capability is
``annotated``: whether the row's own text records its landing. Three landed rows
do not, which is why the plan can honestly report both 75 and 78 landed rows
depending on which question is asked. The ledger carries the distinction as data
so the two numbers are derived rather than re-argued. Delete the field once every
landed row carries its own note.

Usage:
    python tools/parity_plan_ledger.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "contracts" / "parity-pr-plan-ledger.toml"
PYTEST_CONFIG_PATH = REPO_ROOT / "pytest.ini"

EXPECTED_SCHEMA = "flagquantum_parity_pr_plan_ledger_v1"

# The kinds of work a row can deliver, named for what the row produces rather
# than for the row's stage in a workflow.
KIND_VOCABULARY = frozenset(
    {
        "adr",
        "conformance",
        "contract",
        "doc",
        "evidence",
        "fake",
        "impl",
        "tool",
    }
)

SIZE_VOCABULARY = frozenset({"S", "M", "L", "XL"})

STATUS_VOCABULARY = frozenset({"landed", "blocked", "pending"})

# Why a row cannot proceed. Both are conditions the row cannot satisfy by
# writing code: one wants an owner's ruling, the other wants hardware.
BLOCKER_VOCABULARY = frozenset(
    {"owner_decision_required", "accelerator_environment_unavailable"}
)

ROW_ID_PATTERN = re.compile(r"^W(?P<wave>\d+)-(?P<index>\d+)(?P<split>[ab])?$")


def _gate_vocabulary() -> frozenset[str]:
    """The names a ``gate_tier`` entry may use.

    Two real sources rather than a third list: ``tools/ci_tier.py`` owns the tier
    names a workflow runs, and ``pytest.ini`` owns the marker names a lane
    selects. A row may name either, because a row is proved by a tier script or
    by a marker lane, and the plan's own column already mixes the two.
    """

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.ci_tier import CI_TIERS

    markers: set[str] = set()
    in_block = False
    for line in PYTEST_CONFIG_PATH.read_text().splitlines():
        if line.strip().startswith("markers"):
            in_block = True
            continue
        if in_block:
            if not line.strip() or not line.startswith("    "):
                in_block = False
                continue
            markers.add(line.strip().split(":", 1)[0].strip())
    return frozenset(CI_TIERS) | frozenset(markers)


def _load(path: Path = LEDGER_PATH) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def validate(document: dict[str, Any]) -> list[str]:
    """Return the ledger's defects. An empty list means the ledger is sound."""

    problems: list[str] = []
    if document.get("schema") != EXPECTED_SCHEMA:
        problems.append(
            f"ledger schema is {document.get('schema')!r}, expected {EXPECTED_SCHEMA!r}"
        )
        return problems

    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["ledger carries no [[rows]] entries"]

    declared_count = document.get("row_count")
    if declared_count != len(rows):
        problems.append(
            f"row_count is {declared_count!r} but the ledger has {len(rows)} rows"
        )

    gates = _gate_vocabulary()
    identifiers: list[str] = []
    for position, row in enumerate(rows):
        identifier = row.get("id")
        if not isinstance(identifier, str):
            problems.append(f"row {position} has no string id")
            continue
        match = ROW_ID_PATTERN.match(identifier)
        if match is None:
            problems.append(
                f"{identifier}: id does not name a wave, an index, and an optional split"
            )
            continue
        identifiers.append(identifier)

        kinds = row.get("kind")
        if not isinstance(kinds, list) or not kinds:
            problems.append(f"{identifier}: kind must be a non-empty list")
        else:
            for kind in kinds:
                if kind not in KIND_VOCABULARY:
                    problems.append(f"{identifier}: kind {kind!r} is not a work kind")

        tiers = row.get("gate_tier")
        if not isinstance(tiers, list):
            problems.append(f"{identifier}: gate_tier must be a list")
        else:
            for tier in tiers:
                if tier not in gates:
                    problems.append(
                        f"{identifier}: gate_tier {tier!r} is neither a tier name "
                        "nor a pytest marker"
                    )

        if row.get("size") not in SIZE_VOCABULARY:
            problems.append(f"{identifier}: size {row.get('size')!r} is not a row size")

        dependencies = row.get("depends_on")
        if not isinstance(dependencies, list):
            problems.append(f"{identifier}: depends_on must be a list")
            dependencies = []
        for dependency in dependencies:
            if dependency == identifier:
                problems.append(f"{identifier}: depends on itself")
            elif not isinstance(dependency, str) or not ROW_ID_PATTERN.match(
                dependency
            ):
                problems.append(
                    f"{identifier}: dependency {dependency!r} is not a row id"
                )

        status = row.get("status")
        if status not in STATUS_VOCABULARY:
            problems.append(f"{identifier}: status {status!r} is not a row status")
        annotated = row.get("annotated")
        if not isinstance(annotated, bool):
            problems.append(f"{identifier}: annotated must be a boolean")
        elif annotated and status == "pending":
            problems.append(
                f"{identifier}: a pending row records no landing note, so it cannot be "
                "annotated; a row with a landing note is either landed or blocked"
            )
        blocker = row.get("blocker")
        if status == "blocked":
            if blocker not in BLOCKER_VOCABULARY:
                problems.append(
                    f"{identifier}: a blocked row must name a blocker, got {blocker!r}"
                )
        elif blocker is not None:
            problems.append(
                f"{identifier}: a {status} row must not name a blocker, got {blocker!r}"
            )

    duplicated = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    for identifier in duplicated:
        problems.append(f"{identifier}: id appears more than once")

    known = set(identifiers)
    position_of = {identifier: index for index, identifier in enumerate(identifiers)}
    for index, row in enumerate(rows):
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in known:
            continue
        for dependency in row.get("depends_on") or ():
            if dependency not in known:
                problems.append(
                    f"{identifier}: depends on {dependency}, which no row declares"
                )
            elif position_of[dependency] >= index:
                problems.append(
                    f"{identifier}: depends on {dependency}, which the ledger lists at or "
                    "after it; the dependency column orders rows and cannot point forward"
                )
    problems.extend(_cycle_problems(rows, known))
    return problems


def _cycle_problems(rows: list[dict[str, Any]], known: set[str]) -> list[str]:
    edges = {
        row["id"]: [d for d in (row.get("depends_on") or ()) if d in known]
        for row in rows
        if isinstance(row.get("id"), str) and row["id"] in known
    }
    settled: set[str] = set()
    while True:
        ready = {node for node, deps in edges.items() if set(deps) <= settled}
        grown = ready - settled
        if not grown:
            break
        settled |= grown
    unresolved = sorted(set(edges) - settled)
    if not unresolved:
        return []
    return [f"dependency cycle among {len(unresolved)} rows: {unresolved}"]


def _wave_of(identifier: str) -> str:
    match = ROW_ID_PATTERN.match(identifier)
    assert match is not None, identifier
    return match.group("wave")


def summarize(document: dict[str, Any]) -> str:
    rows = document["rows"]
    statuses = Counter(row["status"] for row in rows)
    blockers = Counter(row["blocker"] for row in rows if "blocker" in row)
    annotated = sum(1 for row in rows if row["status"] == "landed" and row["annotated"])
    silent = [
        row["id"] for row in rows if row["status"] == "landed" and not row["annotated"]
    ]
    waves = Counter(_wave_of(row["id"]) for row in rows)
    return (
        f"parity plan ledger: {len(rows)} rows across {len(waves)} waves "
        f"({', '.join(f'W{wave}:{count}' for wave, count in sorted(waves.items()))})\n"
        f"  landed: {statuses['landed']} "
        f"({annotated} carry their own landing note, {len(silent)} do not: "
        f"{', '.join(silent)})\n"
        f"  blocked: {statuses['blocked']} "
        f"({', '.join(f'{name}:{count}' for name, count in sorted(blockers.items()))})\n"
        f"  pending: {statuses['pending']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and exit non-zero on any defect (the default behaviour)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="read this ledger instead of contracts/parity-pr-plan-ledger.toml",
    )
    args = parser.parse_args(argv)

    document = _load(args.ledger)
    problems = validate(document)
    if problems:
        for problem in problems:
            print(f"parity-plan-ledger: {problem}", file=sys.stderr)
        return 1
    print(summarize(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
