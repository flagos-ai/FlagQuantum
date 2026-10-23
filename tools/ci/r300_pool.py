"""Plan deterministic workflow routes using only explicitly qualified R300 sets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).with_suffix(".json")
LABEL = re.compile(r"flagcicd-r300-cpu(?:0[1-9]|1[0-6]|115)")


def load_pool(path: Path = REGISTRY) -> dict[str, Any]:
    """Reject invalid membership and missing qualification evidence."""
    pool: dict[str, Any] = json.loads(path.read_text())
    if pool.get("schema") != 1:
        raise ValueError("Unsupported R300 pool schema")
    seen: set[str] = set()
    for member in pool["members"]:
        label = member["label"]
        if not isinstance(label, str) or not LABEL.fullmatch(label) or label in seen:
            raise ValueError(f"Invalid or duplicate R300 target: {label!r}")
        seen.add(label)
        if member["state"] not in {"qualified", "candidate", "quarantined"}:
            raise ValueError(f"Invalid admission state for {label}")
        if member["state"] == "qualified" and not str(
            member.get("evidence", "")
        ).startswith("https://github.com/flagos-ai/FlagQuantum/actions/runs/"):
            raise ValueError(f"Missing qualification evidence for {label}")
    batch = pool["qualification_batch"]
    if (
        not isinstance(batch, list)
        or len(batch) > 4
        or len(set(batch)) != len(batch)
        or not set(batch) <= seen
    ):
        raise ValueError("Qualification batch must name at most four known targets")
    jobs = pool["jobs"]
    if (
        not jobs
        or len(set(jobs)) != len(jobs)
        or any(
            not isinstance(job, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", job)
            for job in jobs
        )
    ):
        raise ValueError("Invalid or duplicate R300 job names")
    return pool


def plan(pool: dict[str, Any], seed: int) -> dict[str, str]:
    """Spread this run's jobs across distinct qualified sets; never guess health."""
    labels = [m["label"] for m in pool["members"] if m["state"] == "qualified"]
    if seed < 0 or len(labels) < len(pool["jobs"]):
        raise ValueError(
            "A nonnegative seed and distinct qualified targets are required"
        )
    return {
        job: labels[(seed + index) % len(labels)]
        for index, job in enumerate(pool["jobs"])
    }


def qualification_targets(pool: dict[str, Any], limit: int) -> list[str]:
    """Select a bounded candidate batch without admitting it to production."""
    if not 1 <= limit <= 4:
        raise ValueError("Qualification batches must contain at most four targets")
    candidates = {m["label"] for m in pool["members"] if m["state"] == "candidate"}
    return [label for label in pool["qualification_batch"] if label in candidates][
        :limit
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--qualification-limit", type=int, default=4)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    pool = load_pool(args.registry)
    routes = plan(pool, args.seed)
    candidates = qualification_targets(pool, args.qualification_limit)
    print(
        json.dumps(
            {"routes": routes, "qualification_targets": candidates}, sort_keys=True
        )
    )
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f'r300-routes={json.dumps(routes, separators=(",", ":"))}\n')
            output.write(f"qualification-targets={json.dumps(candidates)}\n")


if __name__ == "__main__":
    main()
