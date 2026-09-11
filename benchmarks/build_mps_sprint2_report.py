"""Build the MPS Sprint-2 frozen-matrix closeout report."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_summarize = import_module(
    "benchmarks.build_mps_crossover_report"
)._summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--eight-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    four = _summarize(args.four_gpu)
    eight = _summarize(args.eight_gpu)
    speedup = four["mean_seconds"] / eight["mean_seconds"]
    payload = {
        "schema": "flagquantum.mps_sprint2_closeout.v1",
        "benchmark": "mps_sprint2_frozen_matrix_closeout",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": 8,
        "local_world_size": 8,
        "node_count": 1,
        "rank_placement": eight["rank_placement"],
        "local_memory_bytes_by_rank": eight["local_memory_bytes_by_rank"],
        "communication_bytes": eight["message_totals_by_payload_class"][
            "boundary_forward_tensor"
        ]["payload_bytes"],
        "workload": {
            "n_wires": 64,
            "initial_bond": 32,
            "max_bond": 64,
            "optimizer": "adam",
            "ownership_policy": "communication_aware",
            "prefetch_layer_halos": True,
            "packed_boundary_transport": True,
            "gradient_buckets": True,
            "scaling_mode": "fixed_problem_strong_scaling",
        },
        "points": [four, eight],
        "four_to_eight_speedup": speedup,
        "target_speedup": 1.5,
        "performance_gate_passed": speedup >= 1.5,
        "correctness_gate_passed": (
            four["all_losses_finite"]
            and eight["all_losses_finite"]
            and four["phase_reconciliation_passed"]
            and eight["phase_reconciliation_passed"]
        ),
        "next_phase": "sprint3_large_bond_kernels_and_repeated_statistics",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "four_to_eight_speedup_below_target",
            "single_trial_without_confidence_interval",
            "unsigned_development_evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
