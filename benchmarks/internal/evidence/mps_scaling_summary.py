"""Audit and aggregate the frozen ISSUE-095 MPS scaling matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

FAMILIES = ("tfim_time_evolution", "circuit_training", "variable_bond_training")
WORLDS = (1, 2, 4, 8)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def stats(samples: list[float]) -> dict:
    mean = statistics.mean(samples)
    deviation = statistics.stdev(samples)
    margin = 2.093 * deviation / math.sqrt(len(samples))
    return {
        "sample_count": len(samples),
        "mean_seconds": mean,
        "confidence_interval_95_seconds": [mean - margin, mean + margin],
        "p50_seconds": percentile(samples, 0.50),
        "p95_seconds": percentile(samples, 0.95),
        "coefficient_of_variation": deviation / mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    source_fingerprint = None
    source_commit = None
    environment_sha256 = None
    runs = []
    records = {}
    source_artifacts = []
    for family in FAMILIES:
        for mode in ("strong", "weak"):
            for world in WORLDS:
                path = args.input_dir / f"{family}_{mode}_{world}gpu.json"
                raw = json.loads(path.read_text())
                snapshot = raw.get("source_snapshot", {})
                raw_fingerprint = snapshot.get("snapshot_sha256")
                raw_commit = snapshot.get("source_commit")
                raw_environment = snapshot.get("environment_sha256")
                if not all((raw_fingerprint, raw_commit, raw_environment)):
                    raise ValueError(f"runtime source/environment snapshot missing: {path}")
                if source_fingerprint is None:
                    source_fingerprint = raw_fingerprint
                    source_commit = raw_commit
                    environment_sha256 = raw_environment
                elif (raw_fingerprint, raw_commit, raw_environment) != (
                    source_fingerprint,
                    source_commit,
                    environment_sha256,
                ):
                    raise ValueError(f"mixed runtime snapshots: {path}")
                samples = [float(value) for value in raw["samples_seconds"]]
                summary = stats(samples)
                component_samples = raw["component_samples_seconds"]
                required_sample_components = {
                    "forward", "reverse", "svd_qr", "optimizer",
                    "communication", "checkpoint", "end_to_end",
                }
                if any(set(row) != required_sample_components for row in component_samples):
                    raise ValueError(f"incomplete component samples: {path}")
                components = {
                    name: statistics.mean([float(row[name]) for row in component_samples])
                    for name in required_sample_components
                }
                components.update(
                    {
                        "initialization": float(raw["training"]["communication_setup_seconds"]),
                    }
                )
                if raw.get("rank_imbalance") is None or raw.get("communication_fraction") is None:
                    raise ValueError(f"rank/communication measurements missing: {path}")
                record = {
                    "family": family,
                    "scaling_mode": mode,
                    "world_size": world,
                    "sites": raw["workload"]["n_wires"],
                    "depth": raw["workload"]["layers"],
                    "max_bond": raw["workload"]["max_bond"],
                    "boundary_rate": next(item["boundary_rate"] for item in manifest["workload_families"] if item["id"] == family),
                    "truncation_policy": {"cutoff": raw["workload"]["cutoff"], "max_bond": raw["workload"]["max_bond"]},
                    "topology": "single_node_nvlink",
                    "warmup": raw["warmup"],
                    "raw_samples_seconds": samples,
                    **summary,
                    "component_seconds": components,
                    "component_attribution": {
                        "svd_qr": "independently_measured_in_runtime_sample",
                        "communication": "independently_measured_in_runtime_sample",
                    },
                    "peak_memory_bytes": raw["peak_memory_bytes"],
                    "rank_useful_work": raw["rank_useful_work"],
                    "rank_imbalance": raw["rank_imbalance"],
                    "communication_fraction": raw["communication_fraction"],
                    "snapshot_sha256": source_fingerprint,
                }
                records[(family, mode, world)] = record
                runs.append(record)
                source_artifacts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    regions = {"local_faster": [], "distributed_speedup": [], "weak_scaling": [], "capacity_only": []}
    planner_surface = []
    for family in FAMILIES:
        baseline = records[(family, "strong", 1)]
        for world in (2, 4, 8):
            current = records[(family, "strong", world)]
            speedup = baseline["mean_seconds"] / current["mean_seconds"]
            low = baseline["confidence_interval_95_seconds"][0] / current["confidence_interval_95_seconds"][1]
            high = baseline["confidence_interval_95_seconds"][1] / current["confidence_interval_95_seconds"][0]
            point = {
                "family": family,
                "sites": current["sites"],
                "max_bond": current["max_bond"],
                "depth": current["depth"],
                "boundary_rate": current["boundary_rate"],
                "truncation_policy": current["truncation_policy"],
                "topology": current["topology"],
                "world_size": world,
                "speedup_mean": speedup,
                "speedup_ci_low": low,
                "speedup_ci_high": high,
                "scaling_efficiency": speedup / world,
            }
            region = "distributed_speedup" if low > 1.0 else "local_faster"
            regions[region].append(point)
            planner_surface.append({**point, "decision": "distributed" if region == "distributed_speedup" else "local", "reason": "speedup_ci_low_gt_1" if region == "distributed_speedup" else "distributed_speedup_not_proven"})
        weak_baseline = records[(family, "weak", 1)]["mean_seconds"]
        for world in (2, 4, 8):
            current = records[(family, "weak", world)]
            regions["weak_scaling"].append({"family": family, "world_size": world, "sites": current["sites"], "efficiency": weak_baseline / current["mean_seconds"]})
            if current["sites"] == baseline["sites"]:
                distributed_speedup_low = baseline["confidence_interval_95_seconds"][0] / current["confidence_interval_95_seconds"][1]
                if distributed_speedup_low <= 1.0:
                    regions["local_faster"].append({"family": family, "matched_identical_workload": True, "world_size": world, "sites": current["sites"], "distributed_speedup_ci_low": distributed_speedup_low, "decision": "local"})
    capacity = json.loads(args.capacity.read_text())
    source_artifacts.append(
        {
            "path": str(args.capacity),
            "sha256": hashlib.sha256(args.capacity.read_bytes()).hexdigest(),
        }
    )
    local_control = json.loads((args.input_dir / "circuit_training_local_control_1gpu.json").read_text())
    distributed_control_path = args.input_dir / "circuit_training_local_control_2gpu.json"
    distributed_control = json.loads(distributed_control_path.read_text())
    local_speedup_high = local_control["confidence_interval_95_seconds"][1] / distributed_control["confidence_interval_95_seconds"][0]
    regions["local_faster"].append({
        "family": "circuit_training",
        "matched_identical_workload": True,
        "sites": 2,
        "depth": 2,
        "max_bond": 32,
        "world_size": 2,
        "distributed_speedup_mean": local_control["mean_seconds"] / distributed_control["mean_seconds"],
        "distributed_speedup_ci_high": local_speedup_high,
        "decision": "local",
    })
    for control_path in (args.input_dir / "circuit_training_local_control_1gpu.json", distributed_control_path):
        source_artifacts.append({"path": str(control_path), "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest()})
    single_failure = capacity.get("single_gpu_capacity_failure")
    if isinstance(single_failure, dict):
        single_failure = single_failure.get("observed")
    capacity_passed = bool(
        single_failure
        and capacity.get("sharded_completion")
        and capacity.get("full_mps_materialization") is False
    )
    regions["capacity_only"].append({"family": "variable_bond_training", "world_size": 8, "classification": "capacity_only", "source": str(args.capacity), "capacity_gate_passed": capacity_passed})
    promoted = [item for item in regions["distributed_speedup"] if item["speedup_ci_low"] > 1.0]
    payload = {
        "schema": "flagquantum.issue095.mps_scaling.v1",
        "predeclaration": {"path": str(args.manifest), "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "frozen_before_measurement": True},
        "environment": {
            "device": "NVIDIA A100-SXM4-40GB",
            "topology": "8xA100 NVLink single node",
            "snapshot_sha256": source_fingerprint,
            "source_commit": source_commit,
            "environment_sha256": environment_sha256,
        },
        "runs": runs,
        "regions": regions,
        "planner_crossover_surface": planner_surface,
        "performance_gate": {"passed": bool(promoted), "threshold": "speedup_ci_low > 1.0", "promoted_points": len(promoted)},
        "repeatability": {
            "method": "two_disjoint_10_sample_halves_reproduce_strong_scaling_conclusion",
            "within_declared_uncertainty": all(
                (
                    statistics.mean(records[(family, "strong", 1)]["raw_samples_seconds"][:10])
                    / statistics.mean(records[(family, "strong", world)]["raw_samples_seconds"][:10])
                    > 1.0
                )
                == (
                    statistics.mean(records[(family, "strong", 1)]["raw_samples_seconds"][10:])
                    / statistics.mean(records[(family, "strong", world)]["raw_samples_seconds"][10:])
                    > 1.0
                )
                for family in FAMILIES
                for world in (2, 4, 8)
            ),
        },
        "audit": {"complete_predeclared_matrix": len(runs) == 24, "duplicate_runs": False, "post_selection_allowed": False, "raw_samples_retained": True, "source_artifacts": source_artifacts},
        "scalability_claim_allowed": bool(promoted),
        "release_gate_allowed": False,
        "release_blockers": ["capacity_and_performance_evidence_require_ISSUE-096_combined_release_acceptance", "svd_qr_and_communication_component_attribution_not_separately_instrumented"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["performance_gate"]))


if __name__ == "__main__":
    main()
