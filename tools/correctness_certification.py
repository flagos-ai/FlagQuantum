#!/usr/bin/env python3
"""Generate the machine-readable correctness foundation and phase gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flagquantum.testing import TolerancePolicy, certification_matrix

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/correctness_certification.json"


def render() -> str:
    policy = TolerancePolicy()
    payload = {
        "schema_version": "flagquantum_correctness_certification_v1",
        "cases": [case.__dict__ for case in certification_matrix()],
        "tolerances": [policy.__dict__],
        "hardware_lanes": {
            "one_gpu": "local numerical parity",
            "two_gpu": "required true-sharding semantics",
            "four_to_eight_gpu": "scheduled topology, partition and collective behavior",
        },
        "phase_exit_gates": {
            "operator_or_backend_change": ["generated_operator_backend_matrix"],
            "distributed_change": ["cpu_semantics", "healthy_required_local_gpu_lane"],
            "production_release": ["ISSUE-062_release_wide_certification"],
        },
        "claim_boundary": "foundation_only_not_release_certification",
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            raise SystemExit("correctness certification manifest is stale")
    else:
        OUTPUT.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
