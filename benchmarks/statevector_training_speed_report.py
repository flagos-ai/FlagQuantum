#!/usr/bin/env python3
"""Build a fail-closed matched training-speed report with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flagquantum.benchmarking.statevector_scaling_report import (  # noqa: E402
    bootstrap_median_speedup,
)


def build_report(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    identity_fields = (
        "n_wires",
        "name",
        "layers",
        "gate_count",
        "active_wire_count",
        "parameter_count",
        "entanglement",
        "initialization",
        "seed",
        "dtype",
        "optimizer",
        "learning_rate",
        "observable",
    )
    mismatches = [
        key
        for key in identity_fields
        if baseline.get("workload", {}).get(key)
        != candidate.get("workload", {}).get(key)
    ]
    blockers: list[str] = []
    if mismatches:
        blockers.append(f"workload_mismatch:{','.join(mismatches)}")
    if baseline.get("world_size") != 1:
        blockers.append("baseline_must_use_one_gpu")
    if int(candidate.get("world_size", 0)) <= 1:
        blockers.append("candidate_must_be_distributed")
    if baseline.get("warmup") < 5 or candidate.get("warmup") < 5:
        blockers.append("insufficient_warmup")
    if baseline.get("repetitions") < 30 or candidate.get("repetitions") < 30:
        blockers.append("insufficient_samples")
    if not baseline.get("correctness", {}).get("passed") or not candidate.get(
        "correctness", {}
    ).get("passed"):
        blockers.append("correctness_failed")
    if candidate.get("correctness", {}).get("reference_world_size") != 1:
        blockers.append("candidate_not_checked_against_baseline")

    phases = {}
    for phase in ("differentiable_forward", "backward", "optimizer", "end_to_end"):
        phases[phase] = bootstrap_median_speedup(
            baseline[phase]["samples_seconds"],
            candidate[phase]["samples_seconds"],
        )
    statistically_significant = phases["end_to_end"]["confidence_interval"][0] > 1.0
    if not statistically_significant:
        blockers.append("end_to_end_confidence_interval_includes_no_improvement")
    return {
        "schema": "flagquantum.statevector.training_speed_report.v1",
        "artifact_class": "derived_development_report",
        "passed": not blockers,
        "blockers": blockers,
        "release_gate_allowed": False,
        "workload": {
            key: baseline["workload"][key]
            for key in identity_fields
            if key in baseline["workload"]
        },
        "baseline_world_size": baseline["world_size"],
        "candidate_world_size": candidate["world_size"],
        "warmup": baseline["warmup"],
        "samples": baseline["repetitions"],
        "phases": phases,
        "correctness": {
            "reference_value_absolute_error": candidate["correctness"][
                "reference_value_absolute_error"
            ],
            "reference_gradient_absolute_error_max": candidate["correctness"][
                "reference_gradient_absolute_error_max"
            ],
        },
        "peak_memory_bytes": {
            "baseline": baseline["end_to_end"]["peak_memory_bytes_max"],
            "candidate_per_rank": candidate["end_to_end"]["peak_memory_bytes_max"],
        },
        "statistically_significant_end_to_end_speedup": statistically_significant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
