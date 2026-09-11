"""Build a compact 4/8-GPU fixed-problem MPS crossover diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["workload_manifest"].get("scaling_mode") != (
        "fixed_problem_strong_scaling"
    ):
        raise ValueError(f"{path} is not fixed-problem strong-scaling evidence")
    by_step: dict[int, list[dict[str, Any]]] = {}
    for sample in payload["rank_step_samples"]:
        if sample["sample_class"] == "warm":
            by_step.setdefault(int(sample["step"]), []).append(sample)
    end_to_end = [
        max(float(item["end_to_end_seconds"]) for item in values)
        for _, values in sorted(by_step.items())
    ]
    phases = {
        phase: statistics.mean(
            max(float(item["phase_seconds"][phase]) for item in values)
            for _, values in sorted(by_step.items())
        )
        for phase in ("forward", "reverse_vjp", "optimizer", "unattributed")
    }
    warmup = int(payload["warmup_steps"])
    imbalance = max(
        float(value["max_over_min"])
        for step, value in payload["rank_imbalance_by_step"].items()
        if int(step) >= warmup
    )
    largest_bond = max(
        int(step["largest_realized_bond"])
        for rank in payload["rank_runtime_summaries"]
        for step in rank["step_metrics"]
    )
    local_memory = [
        max(int(step["peak_memory_bytes"]) for step in rank["step_metrics"])
        for rank in payload["rank_runtime_summaries"]
    ]
    return {
        "world_size": int(payload["world_size"]),
        "mean_seconds": statistics.mean(end_to_end),
        "p50_seconds": statistics.median(end_to_end),
        "sample_count": len(end_to_end),
        "phase_mean_seconds": phases,
        "maximum_rank_imbalance": imbalance,
        "largest_realized_bond": largest_bond,
        "message_totals_by_payload_class": payload[
            "message_totals_by_payload_class"
        ],
        "compile_setup_seconds_max": float(payload["compile_setup_seconds_max"]),
        "local_memory_bytes_by_rank": local_memory,
        "rank_placement": payload["environment_manifest"]["rank_placement"],
        "source_commit": payload["environment_manifest"]["source_commit"],
        "source_tree_dirty": payload["environment_manifest"]["source_tree_dirty"],
        "all_losses_finite": payload["all_losses_finite"],
        "phase_reconciliation_passed": payload["phase_reconciliation_passed"],
        "profiled_communication_seconds": sum(
            float(rank["profiled_communication_seconds"])
            for rank in payload["rank_runtime_summaries"]
        ),
        "artifact_sha256": _sha(path),
    }


def build_report(four_gpu: Path, eight_gpu: Path) -> dict[str, Any]:
    four = _summarize(four_gpu)
    eight = _summarize(eight_gpu)
    if (four["world_size"], eight["world_size"]) != (4, 8):
        raise ValueError("crossover report requires ordered 4-GPU and 8-GPU inputs")
    speedup = four["mean_seconds"] / eight["mean_seconds"]
    forward_ratio = (
        eight["phase_mean_seconds"]["forward"]
        / four["phase_mean_seconds"]["forward"]
    )
    reverse_ratio = (
        eight["phase_mean_seconds"]["reverse_vjp"]
        / four["phase_mean_seconds"]["reverse_vjp"]
    )
    four_boundary = four["message_totals_by_payload_class"][
        "boundary_forward_tensor"
    ]
    eight_boundary = eight["message_totals_by_payload_class"][
        "boundary_forward_tensor"
    ]
    blockers = []
    if speedup < 1.0:
        blockers.append("eight_gpu_slower_than_four_gpu")
    blockers.append("nccl_visible_transport_breakdown_not_captured")
    return {
        "schema": "flagquantum.mps_crossover_sprint1_report.v1",
        "benchmark": "mps_fixed_problem_crossover_diagnosis",
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
        "communication_bytes": int(eight_boundary["payload_bytes"]),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": tuple(blockers),
        "workload": {
            "n_wires": 64,
            "initial_bond": 32,
            "max_bond": 64,
            "dtype": "complex64",
            "optimizer": "adam",
            "warmup_steps": 5,
            "retained_warm_steps": 20,
            "scaling_mode": "fixed_problem_strong_scaling",
        },
        "points": (four, eight),
        "four_to_eight_speedup": speedup,
        "eight_over_four_phase_ratio": {
            "forward": forward_ratio,
            "reverse_vjp": reverse_ratio,
        },
        "boundary_forward_message_ratio": (
            int(eight_boundary["message_count"])
            / int(four_boundary["message_count"])
        ),
        "boundary_forward_byte_ratio": (
            int(eight_boundary["payload_bytes"])
            / int(four_boundary["payload_bytes"])
        ),
        "diagnosis": {
            "rank_imbalance_primary": eight["maximum_rank_imbalance"] > 1.05,
            "boundary_growth_observed": (
                int(eight_boundary["message_count"])
                > int(four_boundary["message_count"])
            ),
            "primary_optimization_target": (
                "boundary_transport_and_partition_synchronization"
            ),
            "next_actions": (
                "capture NCCL-visible communication labels",
                "pack boundary envelopes across compatible transfers",
                "overlap independent site work with boundary transport",
                "repeat the frozen matrix without changing thresholds",
            ),
        },
        "blockers": tuple(blockers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--eight-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.four_gpu, args.eight_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "four_to_eight_speedup": report["four_to_eight_speedup"],
                "blockers": report["blockers"],
            }
        )
    )


if __name__ == "__main__":
    main()
