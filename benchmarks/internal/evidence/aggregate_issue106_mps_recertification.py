"""Aggregate ISSUE-106 without promoting a cherry-picked performance region."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


CASES = ("small_latency", "crossover", "large_bond")
WORLDS = (1, 2, 4, 8)
T95_20 = 2.093
MATERIAL_IMPROVEMENT = 1.05
LOCAL_REGRESSION_LIMIT = 1.05
PEAK_MEMORY_REGRESSION_LIMIT = 1.05


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(samples: list[float]) -> dict[str, object]:
    mean = statistics.mean(samples)
    deviation = statistics.stdev(samples)
    margin = T95_20 * deviation / math.sqrt(len(samples))
    ordered = sorted(samples)
    return {
        "sample_count": len(samples),
        "mean_seconds": mean,
        "confidence_interval_95_seconds": [mean - margin, mean + margin],
        "p50_seconds": statistics.median(samples),
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "coefficient_of_variation": deviation / mean,
    }


def _step_maxima(report: dict, field: str = "end_to_end_seconds") -> list[float]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for sample in report["rank_step_samples"]:
        if sample["sample_class"] != "warm":
            continue
        value = sample[field] if field in sample else sample["phase_seconds"][field]
        by_step[int(sample["step"])].append(float(value))
    return [max(by_step[step]) for step in sorted(by_step)]


def _peak_memory(report: dict) -> int | None:
    if "rank_runtime_summaries" not in report:
        return None
    return max(
        int(step.get("peak_memory_bytes", 0))
        for rank in report["rank_runtime_summaries"]
        for step in rank["step_metrics"]
        if int(step["step"]) >= int(report["warmup_steps"])
    )


def _comparison(baseline: dict, current: dict) -> dict[str, object]:
    old_stats = _stats(_step_maxima(baseline))
    new_stats = _stats(_step_maxima(current))
    old_low, old_high = old_stats["confidence_interval_95_seconds"]
    new_low, new_high = new_stats["confidence_interval_95_seconds"]
    speedup = old_stats["mean_seconds"] / new_stats["mean_seconds"]
    low = old_low / new_high
    high = old_high / new_low
    phase_speedups = {}
    for phase in ("forward", "reverse_vjp", "optimizer", "unattributed"):
        old = statistics.mean(_step_maxima(baseline, phase))
        new = statistics.mean(_step_maxima(current, phase))
        phase_speedups[phase] = old / new if new else None
    warm_samples = [
        sample for sample in current["rank_step_samples"]
        if sample["sample_class"] == "warm"
    ]
    messages = {
        name: {
            "message_count": sum(int(sample["message_classes"][name]["message_count"]) for sample in warm_samples),
            "payload_bytes": sum(int(sample["message_classes"][name]["payload_bytes"]) for sample in warm_samples),
        }
        for name in current["message_totals_by_payload_class"]
    }
    warm_imbalance = [
        value for step, value in current["rank_imbalance_by_step"].items()
        if int(step) >= int(current["warmup_steps"])
    ]
    profiled_cuda = sum(float(rank["profiled_cuda_activity_seconds"]) for rank in current["rank_runtime_summaries"])
    profiled_comm = sum(float(rank["profiled_communication_seconds"]) for rank in current["rank_runtime_summaries"])
    baseline_peak = _peak_memory(baseline)
    current_peak = _peak_memory(current)
    return {
        "case": current["workload_manifest"]["case"],
        "world_size": current["world_size"],
        "n_wires": current["workload_manifest"]["n_wires"],
        "baseline_end_to_end": old_stats,
        "optimized_end_to_end": new_stats,
        "speedup_mean": speedup,
        "speedup_ci_low": low,
        "speedup_ci_high": high,
        "material_improvement_threshold": MATERIAL_IMPROVEMENT,
        "performance_gate_passed": low > 1.0 and speedup >= MATERIAL_IMPROVEMENT,
        "phase_speedup": phase_speedups,
        "scaling_efficiency_vs_issue097": speedup / int(current["world_size"]),
        "profiled_cuda_activity_seconds": profiled_cuda,
        "profiled_utilization_proxy": min(1.0, profiled_cuda / max(sum(_step_maxima(current)) * int(current["world_size"]), 1e-12)),
        "profiled_communication_fraction": profiled_comm / max(profiled_cuda, 1e-12),
        "warm_message_totals_by_payload_class": messages,
        "warm_rank_imbalance": {
            "maximum_max_over_min": max(float(item["max_over_min"]) for item in warm_imbalance),
            "mean_idle_imbalance_seconds": statistics.mean(float(item["idle_imbalance_seconds"]) for item in warm_imbalance),
        },
        "baseline_peak_memory_bytes": baseline_peak,
        "optimized_peak_memory_bytes": current_peak,
        "peak_memory_ratio": current_peak / max(baseline_peak, 1) if baseline_peak is not None else None,
        "compile_setup_seconds_max": current["compile_setup_seconds_max"],
        "compile_amortization_fraction": current["compile_amortization_fraction"],
    }


def aggregate(current_dir: Path, baseline_dir: Path, root: Path) -> dict[str, object]:
    expected = {(case, world) for case in CASES for world in WORLDS}
    current: dict[tuple[str, int], dict] = {}
    baseline: dict[tuple[str, int], dict] = {}
    artifacts = []
    for directory, destination in ((current_dir, current), (baseline_dir, baseline)):
        for path in sorted(directory.glob("*gpu.json")):
            raw = json.loads(path.read_text())
            key = (raw["workload_manifest"]["case"], int(raw["world_size"]))
            destination[key] = raw
            artifacts.append({"path": str(path), "sha256": _sha(path)})
    blockers = []
    if set(current) != expected or set(baseline) != expected:
        blockers.append("incomplete_predeclared_matrix")
    if blockers:
        return {"schema": "flagquantum.issue106.mps_recertification.v1", "blockers": blockers, "acceptance_passed": False}
    snapshots = {
        json.dumps(report["environment_manifest"]["critical_file_sha256"], sort_keys=True)
        for report in current.values()
    }
    if len(snapshots) != 1:
        blockers.append("optimized_source_snapshot_mismatch")
    if any(report["warm_sample_count_across_ranks"] != 20 * report["world_size"] for report in current.values()):
        blockers.append("twenty_warm_samples_per_rank_missing")
    if any(not report["phase_reconciliation_passed"] for report in current.values()):
        blockers.append("phase_reconciliation_failed")
    if any(not report["all_losses_finite"] for report in current.values()):
        blockers.append("non_finite_loss")
    if any(
        rank["distribution_semantics"] != (
            "single_device_fast_path" if report["world_size"] == 1 else "sharded_across_ranks"
        )
        for report in current.values() for rank in report["rank_runtime_summaries"]
    ):
        blockers.append("non_sharded_runtime_semantics")
    if any(
        len(report.get("trace_paths", ())) != report["world_size"]
        or any(not (root / path).is_file() for path in report.get("trace_paths", ()))
        for report in current.values()
    ):
        blockers.append("optimized_rank_trace_missing")

    baseline_audit_path = root / "benchmarks/development/issue097_mps_critical_path_baseline_recertified.json"
    baseline_audit = json.loads(baseline_audit_path.read_text())
    if not baseline_audit["baseline_gate_passed"]:
        blockers.append("issue097_baseline_audit_failed")

    points = [_comparison(baseline[key], current[key]) for key in sorted(expected)]
    promoted = [point for point in points if point["performance_gate_passed"]]
    correctness_path = root / "benchmarks/development/distributed_mps_correctness_matrix.json"
    capacity_path = root / "benchmarks/development/issue092_general_mps_capacity.json"
    recovery_path = root / "benchmarks/development/issue093_mps_stability_matrix.json"
    correctness = json.loads(correctness_path.read_text())
    capacity = json.loads(capacity_path.read_text())
    recovery = json.loads(recovery_path.read_text())
    gates = {
        "performance": {"passed": bool(promoted), "promoted_points": len(promoted), "threshold": "speedup_ci_low > 1.0 and speedup_mean >= 1.05"},
        "correctness": {"passed": bool(correctness["passed"]), "reference": str(correctness_path.relative_to(root)), "sha256": _sha(correctness_path)},
        "capacity": {"passed": bool(capacity["single_gpu_capacity_failure"] and capacity["sharded_completion"] and not capacity["full_mps_materialization"]), "reference": str(capacity_path.relative_to(root)), "sha256": _sha(capacity_path)},
        "recovery": {"passed": all(item["passed"] and item["cleanup_verified"] for item in recovery["fault_matrix"]), "reference": str(recovery_path.relative_to(root)), "sha256": _sha(recovery_path)},
        "local_fast_path": {"passed": all(point["speedup_mean"] >= 1 / LOCAL_REGRESSION_LIMIT for point in points if point["world_size"] == 1), "maximum_regression_ratio": LOCAL_REGRESSION_LIMIT},
        "peak_memory": {"passed": all(point["peak_memory_ratio"] is not None and point["peak_memory_ratio"] <= PEAK_MEMORY_REGRESSION_LIMIT for point in points), "maximum_regression_ratio": PEAK_MEMORY_REGRESSION_LIMIT, "baseline_available": all(point["baseline_peak_memory_bytes"] is not None for point in points)},
    }
    if not gates["performance"]["passed"]:
        blockers.append("no_predeclared_point_passed_performance_gate")
    for name in ("correctness", "capacity", "recovery", "local_fast_path", "peak_memory"):
        if not gates[name]["passed"]:
            blockers.append(f"{name}_gate_failed")
    decisions = [
        {
            "case": point["case"],
            "sites": point["n_wires"],
            "world_size": point["world_size"],
            "decision": "site_sharded_optimized" if point["performance_gate_passed"] else "local_fast_path",
            "reason": "recertified_optimization_speedup" if point["performance_gate_passed"] else "recertified_speedup_not_proven",
            "speedup_mean": point["speedup_mean"],
            "speedup_ci_low": point["speedup_ci_low"],
            "speedup_ci_high": point["speedup_ci_high"],
        }
        for point in points
    ]
    decisions.append({"case": "issue092_capacity_workload", "world_size": capacity["world_size"], "decision": "capacity_only", "reason": "single_gpu_capacity_failure_and_sharded_completion"})
    representative = current[("crossover", 8)]
    representative_memory = [
        max(int(step.get("peak_memory_bytes", 0)) for step in rank["step_metrics"])
        for rank in representative["rank_runtime_summaries"]
    ]
    representative_communication_bytes = sum(
        int(values["payload_bytes"])
        for values in representative["message_totals_by_payload_class"].values()
    )
    return {
        "schema": "flagquantum.issue106.mps_recertification.v1",
        "predeclared_matrix_complete": True,
        "distribution_semantics": "sharded_across_ranks",
        "matrix_contains_single_device_controls": True,
        "world_size": 8,
        "local_world_size": 8,
        "node_count": 1,
        "rank_placement": representative["environment_manifest"]["rank_placement"],
        "local_memory_bytes_by_rank": representative_memory,
        "communication_bytes": representative_communication_bytes,
        "claim_evidence_type": "development_performance_recertification",
        "optimization_set": "issue098_through_issue105",
        "statistical_method": "per-step rank maximum; two-sided 95% t intervals; conservative ratio bounds",
        "points": points,
        "gates": gates,
        "planner_decisions": decisions,
        "planner_inputs_refreshed": not blockers,
        "planner_refresh_scope": "issue095_issue053_optimization_policy_surface",
        "planner_refresh_reason": "all_acceptance_gates_passed" if not blockers else "all_acceptance_gates_must_pass",
        "source_artifacts": artifacts,
        "baseline_audit": {
            "path": str(baseline_audit_path.relative_to(root)),
            "sha256": _sha(baseline_audit_path),
            "passed": baseline_audit["baseline_gate_passed"],
        },
        "blockers": blockers,
        "acceptance_passed": not blockers,
        "performance_claim_allowed": not blockers,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    payload = aggregate(args.input, args.baseline, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"acceptance_passed": payload["acceptance_passed"], "blockers": payload["blockers"]}))


if __name__ == "__main__":
    main()
