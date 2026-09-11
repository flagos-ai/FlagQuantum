"""Build compact non-release evidence for MPS ownership-policy A/B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _point(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_step: dict[int, list[dict[str, Any]]] = {}
    for sample in payload["rank_step_samples"]:
        if sample["sample_class"] == "warm":
            by_step.setdefault(int(sample["step"]), []).append(sample)
    samples = [
        max(float(item["end_to_end_seconds"]) for item in values)
        for _, values in sorted(by_step.items())
    ]
    boundary = payload["message_totals_by_payload_class"][
        "boundary_forward_tensor"
    ]
    warmup = int(payload["warmup_steps"])
    return {
        "world_size": int(payload["world_size"]),
        "ownership_policy": payload["workload_manifest"]["ownership_policy"],
        "site_ownership": payload["workload_manifest"]["site_ownership"],
        "mean_seconds": statistics.mean(samples),
        "p50_seconds": statistics.median(samples),
        "sample_count": len(samples),
        "boundary_forward_messages": int(boundary["message_count"]),
        "boundary_forward_bytes": int(boundary["payload_bytes"]),
        "compile_setup_seconds_max": float(payload["compile_setup_seconds_max"]),
        "maximum_rank_imbalance": max(
            float(value["max_over_min"])
            for step, value in payload["rank_imbalance_by_step"].items()
            if int(step) >= warmup
        ),
        "all_losses_finite": payload["all_losses_finite"],
        "phase_reconciliation_passed": payload["phase_reconciliation_passed"],
        "source_commit": payload["environment_manifest"]["source_commit"],
        "source_tree_dirty": payload["environment_manifest"]["source_tree_dirty"],
        "rank_placement": payload["environment_manifest"]["rank_placement"],
        "local_memory_bytes_by_rank": [
            max(int(step["peak_memory_bytes"]) for step in rank["step_metrics"])
            for rank in payload["rank_runtime_summaries"]
        ],
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("four_equal", "four_aware", "eight_equal", "eight_aware"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = [
        _point(args.four_equal),
        _point(args.four_aware),
        _point(args.eight_equal),
        _point(args.eight_aware),
    ]
    indexed = {
        (point["world_size"], point["ownership_policy"]): point
        for point in points
    }
    four_equal = indexed[(4, "equal_sites")]
    four_aware = indexed[(4, "communication_aware")]
    eight_equal = indexed[(8, "equal_sites")]
    eight_aware = indexed[(8, "communication_aware")]
    four_gain = four_equal["mean_seconds"] / four_aware["mean_seconds"]
    eight_gain = eight_equal["mean_seconds"] / eight_aware["mean_seconds"]
    strong_speedup = four_aware["mean_seconds"] / eight_aware["mean_seconds"]
    payload = {
        "schema": "flagquantum.mps_ownership_ab_report.v1",
        "benchmark": "mps_communication_aware_ownership_ab",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": 8,
        "local_world_size": 8,
        "node_count": 1,
        "rank_placement": eight_aware["rank_placement"],
        "local_memory_bytes_by_rank": eight_aware[
            "local_memory_bytes_by_rank"
        ],
        "communication_bytes": eight_aware["boundary_forward_bytes"],
        "workload": {
            "n_wires": 64,
            "initial_bond": 32,
            "max_bond": 64,
            "scaling_mode": "fixed_problem_strong_scaling",
        },
        "points": points,
        "four_gpu_policy_speedup": four_gain,
        "eight_gpu_policy_speedup": eight_gain,
        "aware_four_to_eight_speedup": strong_speedup,
        "eight_gpu_boundary_message_reduction": (
            1 - eight_aware["boundary_forward_messages"]
            / eight_equal["boundary_forward_messages"]
        ),
        "eight_gpu_boundary_byte_reduction": (
            1 - eight_aware["boundary_forward_bytes"]
            / eight_equal["boundary_forward_bytes"]
        ),
        "eight_gpu_compile_setup_ratio": (
            eight_aware["compile_setup_seconds_max"]
            / eight_equal["compile_setup_seconds_max"]
        ),
        "passed": (
            all(point["all_losses_finite"] for point in points)
            and all(point["phase_reconciliation_passed"] for point in points)
            and eight_gain > 1
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "single_ab_run_unsigned_development_evidence",
            "aware_four_to_eight_speedup_below_one",
            "communication_aware_compile_setup_regression",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
