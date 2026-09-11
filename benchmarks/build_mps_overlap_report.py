"""Build compact non-release evidence for MPS halo-overlap A/B runs."""

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
    end_to_end = [
        max(float(item["end_to_end_seconds"]) for item in samples)
        for _, samples in sorted(by_step.items())
    ]
    return {
        "world_size": payload["world_size"],
        "prefetch_layer_halos": payload["workload_manifest"][
            "prefetch_layer_halos"
        ],
        "mean_seconds": statistics.mean(end_to_end),
        "p50_seconds": statistics.median(end_to_end),
        "sample_count": len(end_to_end),
        "all_losses_finite": payload["all_losses_finite"],
        "phase_reconciliation_passed": payload["phase_reconciliation_passed"],
        "source_commit": payload["environment_manifest"]["source_commit"],
        "source_tree_dirty": payload["environment_manifest"]["source_tree_dirty"],
        "rank_placement": payload["environment_manifest"]["rank_placement"],
        "local_memory_bytes_by_rank": [
            max(int(step["peak_memory_bytes"]) for step in rank["step_metrics"])
            for rank in payload["rank_runtime_summaries"]
        ],
        "communication_bytes": int(
            payload["message_totals_by_payload_class"][
                "boundary_forward_tensor"
            ]["payload_bytes"]
        ),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("four_off", "four_on", "eight_off", "eight_on"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = [
        _point(args.four_off),
        _point(args.four_on),
        _point(args.eight_off),
        _point(args.eight_on),
    ]
    indexed = {
        (point["world_size"], point["prefetch_layer_halos"]): point
        for point in points
    }
    four_speedup = indexed[(4, False)]["mean_seconds"] / indexed[(4, True)][
        "mean_seconds"
    ]
    eight_speedup = indexed[(8, False)]["mean_seconds"] / indexed[(8, True)][
        "mean_seconds"
    ]
    strong_speedup = indexed[(4, True)]["mean_seconds"] / indexed[(8, True)][
        "mean_seconds"
    ]
    payload = {
        "schema": "flagquantum.mps_halo_overlap_report.v1",
        "benchmark": "mps_halo_overlap_ab",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": 8,
        "local_world_size": 8,
        "node_count": 1,
        "rank_placement": indexed[(8, True)]["rank_placement"],
        "local_memory_bytes_by_rank": indexed[(8, True)][
            "local_memory_bytes_by_rank"
        ],
        "communication_bytes": indexed[(8, True)]["communication_bytes"],
        "workload": {
            "n_wires": 64,
            "initial_bond": 32,
            "max_bond": 64,
            "scaling_mode": "fixed_problem_strong_scaling",
        },
        "points": points,
        "four_gpu_overlap_speedup": four_speedup,
        "eight_gpu_overlap_speedup": eight_speedup,
        "overlap_enabled_four_to_eight_speedup": strong_speedup,
        "passed": (
            all(point["all_losses_finite"] for point in points)
            and all(point["phase_reconciliation_passed"] for point in points)
            and eight_speedup > 1.0
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "single_ab_run_unsigned_development_evidence",
            "four_to_eight_speedup_below_one",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
