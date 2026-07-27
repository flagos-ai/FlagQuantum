#!/usr/bin/env python3
"""Validate the checked-in branch-protection required-check contract."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / ".github/required-checks.json").read_text())
    checks = tuple(payload.get("checks", ()))
    required = {
        "CI / quality",
        "CI / cpu-core (3.10)",
        "CI / cpu-core (3.11)",
        "CI / cpu-core (3.12)",
        "CI / package",
        "CI / distributed-cpu",
        "Local GPU Gate / two-gpu-distributed-required",
    }
    missing = required - set(checks)
    unexpected = set(checks) - required
    if (
        payload.get("schema") != "flagquantum_required_checks_v1"
        or missing
        or unexpected
        or len(checks) != len(set(checks))
    ):
        raise SystemExit(
            "invalid required-check policy; "
            f"missing={sorted(missing)}; unexpected={sorted(unexpected)}"
        )
    ci = (root / ".github/workflows/ci.yml").read_text()
    gpu = (root / ".github/workflows/local-gpu.yml").read_text()
    for job in ("quality:", "package:", "distributed-cpu:"):
        if job not in ci:
            raise SystemExit(f"required workflow job missing: {job[:-1]}")
    if "two-gpu-distributed-required:" not in gpu:
        raise SystemExit("required two-GPU workflow job missing")
    if "paths:" in gpu.split("pull_request:", 1)[1].split("jobs:", 1)[0]:
        raise SystemExit("required two-GPU workflow must emit a check for every PR")
    if "cpu-core:" not in ci or not all(
        f'"{version}"' in ci for version in ("3.10", "3.11", "3.12")
    ):
        raise SystemExit("required Python 3.10-3.12 cpu-core matrix is incomplete")
    print(f"validated {len(checks)} externally configured required checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
