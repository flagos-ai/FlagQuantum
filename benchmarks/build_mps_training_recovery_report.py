"""Report the fixed MPS training series without promoting release claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmarks.internal.evidence.mps_training_recovery import tree_max_error


def trajectory(point: dict[str, Any], field: str) -> list[Any]:
    rows = []
    for step in range(point["protocol"]["steps"]):
        if field == "loss":
            rows.append(
                point["ranks"][0]["uninterrupted"]["step_metrics"][step]["loss"]
            )
        else:
            values = {
                int(i): float(v)
                for rank in point["ranks"]
                for i, v in rank["uninterrupted"]["step_metrics"][step][field]
            }
            if sorted(values) != list(
                range(len(point["protocol"]["initial_parameters"]))
            ):
                raise ValueError("missing owned parameter diagnostic")
            rows.append([values[i] for i in sorted(values)])
    return rows


def validate(point: dict[str, Any], world: int) -> None:
    protocol = point["protocol"]
    if point["status"] != "completed" or point["source"]["dirty"]:
        raise ValueError("completed clean-source measurement required")
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    # The raw protocol digest covers its original file bytes, not canonical JSON.
    if (
        hashlib.sha256(encoded).hexdigest() != point["protocol_canonical_sha256"]
        or not point["source"]["protocol_sha256"]
    ):
        raise ValueError("missing frozen protocol")
    if point["world_size"] != world or len(point["ranks"]) != world:
        raise ValueError("incomplete rank inventory")
    sites = []
    owners = []
    for rank, record in enumerate(point["ranks"]):
        summary = record["uninterrupted"]
        if record["rank"] != rank or summary["world_size"] != world:
            raise ValueError("rank identity mismatch")
        if not record["full_state_gather_forbidden_during_training"]:
            raise ValueError("full-state collection guard missing")
        ownership = summary["site_ownership"]
        if len(ownership) != world:
            raise ValueError("site ownership topology mismatch")
        sites.extend(ownership[rank])
        owned = record["checkpoint_reference"]["owned_indices"]
        owners.extend(owned)
        if (
            owned != record["checkpoint_resumed"]["owned_indices"]
            or owned != record["checkpoint_prefix"]["owned_indices"]
        ):
            raise ValueError("checkpoint owner mismatch")
        reference = summary["step_metrics"]
        restarted = (
            record["restarted_prefix"]["step_metrics"]
            + record["resumed"]["step_metrics"]
        )
        if len(reference) != protocol["steps"] or len(restarted) != len(reference):
            raise ValueError("incomplete consecutive updates")
        if [s["step"] for s in reference] != list(range(protocol["steps"])):
            raise ValueError("nonconsecutive updates")
        if record["resumed"]["start_step"] != protocol["checkpoint_step"]:
            raise ValueError("checkpoint was not resumed at frozen boundary")
        errors = {
            "loss": tree_max_error(
                [s["loss"] for s in reference], [s["loss"] for s in restarted]
            ),
            "gradients": tree_max_error(
                [s["parameter_gradients"] for s in reference],
                [s["parameter_gradients"] for s in restarted],
            ),
            "parameters": tree_max_error(
                record["reference_parameters_final"], record["resumed_parameters_final"]
            ),
            "checkpoint_parameters": tree_max_error(
                record["checkpoint_reference"]["parameters"],
                record["checkpoint_resumed"]["parameters"],
            ),
            "optimizer_state": tree_max_error(
                record["checkpoint_reference"]["optimizer_state"],
                record["checkpoint_resumed"]["optimizer_state"],
            ),
        }
        if (max(errors.values()) <= protocol["comparison_atol"]) != record[
            "recovery_passed"
        ]:
            raise ValueError("recovery comparison label disagrees with raw state")
        for step in reference:
            if (
                not all(
                    math.isfinite(step[key])
                    for key in ("loss", "discarded_weight", "end_to_end_seconds")
                )
                or step["end_to_end_seconds"] <= 0
            ):
                raise ValueError("nonfinite or invalid step sample")
    if sorted(sites) != list(range(protocol["n_sites"])):
        raise ValueError("site owners do not partition one workload")
    if sorted(owners) != list(range(len(protocol["initial_parameters"]))):
        raise ValueError("optimizer owners do not partition parameters")
    if len(point["matched_local_mps"]) != protocol["steps"]:
        raise ValueError("missing matched local MPS validation")


def report(directory: Path, worlds: list[int]) -> tuple[dict[str, Any], str]:
    measurements: dict[int, dict[str, Any]] = {}
    outcomes = []
    digests = {}
    for world in worlds:
        path = directory / f"world-{world}.json"
        if path.exists():
            point = json.loads(path.read_text())
            validate(point, world)
            measurements[world] = point
            digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            receipt = directory / f"world-{world}-outcome.json"
            if not receipt.exists():
                raise ValueError(f"missing deployment outcome: world {world}")
            outcome = json.loads(receipt.read_text())
            if outcome["status"] == "completed":
                raise ValueError("successful worker has no measurement")
            outcomes.append(outcome)
    baseline = measurements.get(1)
    reference = next(iter(measurements.values()), None)
    if reference is None:
        raise ValueError("no completed training point")
    rows = []
    for world, point in measurements.items():
        if (
            point["source"] != reference["source"]
            or point["protocol"] != reference["protocol"]
            or point["backend"] != reference["backend"]
        ):
            raise ValueError(
                "strong scaling requires identical source, workload and device backend"
            )
        samples = [
            max(
                r["uninterrupted"]["step_metrics"][step]["end_to_end_seconds"]
                for r in point["ranks"]
            )
            for step in range(point["protocol"]["steps"])
        ]
        med = statistics.median(samples[1:])
        scaling_errors = {
            field: (
                tree_max_error(trajectory(point, field), trajectory(baseline, field))
                if baseline
                else None
            )
            for field in ("loss", "parameter_gradients", "parameter_values")
        }
        diagnostics = point["matched_local_mps"]
        row = {
            "world_size": world,
            "node_count": point["node_count"],
            "status": "completed",
            "objective_pre_update_samples": trajectory(point, "loss"),
            "step_critical_rank_seconds": samples,
            "statistic": f"median_steps_2_through_{point['protocol']['steps']}",
            "steady_median_seconds": med,
            "scaling_errors_vs_world_1": scaling_errors,
            "matched_local_objective_error": max(
                d["sharding_objective_max_abs_error"] for d in diagnostics
            ),
            "matched_local_projected_gradient_error": max(
                d["sharding_projected_gradient_max_abs_error"] for d in diagnostics
            ),
            "local_derivative_semantics_error": max(
                d["derivative_semantics_max_abs_difference"] for d in diagnostics
            ),
            "target_met_at_last_pre_update_objective": trajectory(point, "loss")[-1]
            <= point["protocol"]["objective_target"],
            "recovery_passed": point["recovery_passed"],
            "per_rank_peak_bytes": [
                max(s["peak_memory_bytes"] for s in r["uninterrupted"]["step_metrics"])
                for r in point["ranks"]
            ],
        }
        rows.append(row)
    baseline_time = next(
        (r["steady_median_seconds"] for r in rows if r["world_size"] == 1), None
    )
    for row in rows:
        row["measured_speedup"] = (
            baseline_time / row["steady_median_seconds"] if baseline_time else None
        )
        row["measured_efficiency"] = (
            row["measured_speedup"] / row["world_size"] if baseline_time else None
        )
    payload = {
        "source": reference["source"],
        "protocol": reference["protocol"],
        "measurement_sha256": digests,
        "rows": rows,
        "unsuccessful_outcomes": outcomes,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "amdahl_estimate": None,
        "claim_evidence_type": "development_smoke",
    }
    lines = [
        "# Fixed-workload MPS training and recovery",
        "",
        "One frozen 32-site, three-layer problem; eight Adam updates. The 131072-site capacity leg is separate. These are development measurements, with no new certified rank count or capability promotion.",
        "",
        "Costs use the maximum recorded rank time for each uninterrupted update; all eight raw samples are retained, and the steady statistic is the median of updates 2–8. Whole-call barrier-bracketed wall time and checkpoint IO are separate in raw rank summaries. The one-rank control uses the same distributed engine, not the fastest local implementation. Local MPS diagnostics are outside these timers.",
        "",
        "Efficiency is measured T1/(p*Tp) on the identical workload and backend, not an Amdahl estimate. No parallel fraction was measured or fitted. CPU/Gloo correctness is separate from CUDA/NCCL hardware evidence.",
        "",
        "| Ranks | Nodes | Initial / last pre-update objective | Median s | Speedup | Efficiency | Local projected-VJP error | Local VJP vs FD | Recovery |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        loss = row["objective_pre_update_samples"]
        speed = (
            f"{row['measured_speedup']:.4f}"
            if row["measured_speedup"] is not None
            else "unavailable"
        )
        efficiency = (
            f"{row['measured_efficiency']:.4f}"
            if row["measured_efficiency"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {row['world_size']} | {row['node_count']} | {loss[0]:.8f} / {loss[-1]:.8f} | {row['steady_median_seconds']:.6f} | {speed} | {efficiency} | {row['matched_local_projected_gradient_error']:.4g} | {row['local_derivative_semantics_error']:.4g} | {row['recovery_passed']} |"
        )
    for outcome in outcomes:
        lines.append(
            f"| {outcome['world_size']} | — | {outcome['status']}: inspect retained diagnostics | — | — | — | — | — | — |"
        )
    lines += [
        "",
        "The gradient is the existing projected stop-subspace approximation. Its difference from finite differences of the truncated forward map is recorded separately from sharding error against the same local approximate MPS. Losses in the table are pre-update, not a post-final-update reevaluation.",
        "",
        "Each rank retains forward/reverse/optimizer/diagnostic timings, allocated/reserved snapshots, allocated peaks, bond/discarded-weight traces and boundary/gradient/optimizer communication counters. Those counters are scoped runtime payload counts, not a packet-level total including every metadata exchange. CPU allocator zeros do not establish zero memory use; worker RSS is separate. Topology and physical inventories accompany accelerator launches.",
        "",
        "Recovery reconstructs parameters and the owner optimizer from committed rank shards at the same world size. The circuit is reevaluated from its fixed product state on every update; a continuously evolving MPS is not checkpointed. No checkpoint rank resize, arbitrary entanglement training, convergence theorem or full-state gather is claimed. The full-state gather and object-gather APIs are guarded inside training; only compact scalar diagnostics are collected afterward.",
        "",
    ]
    return payload, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", default="1,2,4,8,16")
    args = parser.parse_args()
    payload, markdown = report(args.input, [int(v) for v in args.worlds.split(",")])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(markdown)


if __name__ == "__main__":
    main()
