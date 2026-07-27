#!/usr/bin/env python
"""Fail-closed preflight for sealing release-certified runtime evidence."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SCALABILITY_ROOT = ROOT / "benchmarks" / "results" / "scalability"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def readiness_errors(
    *,
    commit: str,
    signing_key_present: bool,
    device_uuids: Sequence[str],
    world_size: int,
    promoted_json: Sequence[Path] = (),
) -> tuple[str, ...]:
    """Return release-environment blockers without exposing secret material."""

    errors: list[str] = []
    if not _COMMIT.fullmatch(commit):
        errors.append("release evidence requires a full 40-character Git commit")
    if not signing_key_present:
        errors.append("FQ_EVIDENCE_SIGNING_KEY is not configured")
    if world_size < 1:
        errors.append("world_size must be positive")
    elif len(tuple(device_uuids)) < world_size:
        errors.append(
            f"release evidence requires {world_size} GPU UUIDs; "
            f"detected {len(tuple(device_uuids))}"
        )
    unexpected = tuple(
        path for path in promoted_json if path.name != "scalability_audit_summary.json"
    )
    if unexpected:
        errors.append(
            "scalability directory already contains promoted JSON; seal into the "
            "candidate directory and promote only after strict audit"
        )
    return tuple(errors)


def _output(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def environment_errors(*, world_size: int) -> tuple[str, ...]:
    commit = _output(("git", "rev-parse", "HEAD"))
    devices = tuple(
        line.strip()
        for line in _output(
            ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
        ).splitlines()
        if line.strip()
    )
    promoted = tuple(SCALABILITY_ROOT.rglob("*.json"))
    return readiness_errors(
        commit=commit,
        signing_key_present=bool(os.environ.get("FQ_EVIDENCE_SIGNING_KEY")),
        device_uuids=devices,
        world_size=world_size,
        promoted_json=promoted,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args(argv)
    errors = environment_errors(world_size=args.world_size)
    if errors:
        for error in errors:
            print(f"BLOCKED: {error}")
        return 2
    print("release evidence environment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
