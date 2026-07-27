#!/usr/bin/env python3
"""Aggregate high-load differentiable statevector weak-scaling artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .statevector_scaling_report import bootstrap_median_speedup
except ImportError:  # direct script execution
    from statevector_scaling_report import bootstrap_median_speedup

SCHEMA = "flagquantum.statevector.training_scaling_report.v1"
SOURCE_SCHEMA = "flagquantum.statevector.training_scaling.v1"
PHASES = ("forward", "differentiable_forward", "backward", "end_to_end")


def build_report(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    by_world = {int(item.get("world_size", 0)): item for item in artifacts}
    if len(by_world) != len(artifacts) or 1 not in by_world:
        raise ValueError("unique world sizes including one required")
    worlds = sorted(by_world)
    if any(world & (world - 1) for world in worlds):
        raise ValueError("power-of-two world sizes required")
    baseline = by_world[1]
    distributed_baseline = by_world.get(2)
    baseline_wires = int(baseline["workload"]["n_wires"])
    local_amplitudes = int(baseline["workload"]["local_amplitudes"])
    for world, item in by_world.items():
        if item.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("invalid source schema")
        if not item.get("correctness", {}).get("passed"):
            raise ValueError("every point must pass correctness")
        workload = item["workload"]
        if int(workload["n_wires"]) != baseline_wires + int(math.log2(world)):
            raise ValueError("invalid weak-scaling qubit progression")
        if int(workload["local_amplitudes"]) != local_amplitudes:
            raise ValueError("local amplitudes must remain constant")
        if len(item.get("rank_placement", [])) != world:
            raise ValueError("incomplete rank placement")
        for phase in PHASES:
            if len(item.get(phase, {}).get("samples_seconds", [])) < 3:
                raise ValueError(f"insufficient {phase} samples")

    points = []
    for world in worlds:
        item = by_world[world]
        phases = {}
        for phase in PHASES:
            samples = item[phase]["samples_seconds"]
            efficiency = (
                {
                    "method": "baseline",
                    "speedup": 1.0,
                    "confidence_level": 0.95,
                    "confidence_interval": [1.0, 1.0],
                }
                if world == 1
                else bootstrap_median_speedup(
                    baseline[phase]["samples_seconds"], samples
                )
            )
            phases[phase] = {
                "median_seconds": item[phase]["median_seconds"],
                "coefficient_of_variation": item[phase]["coefficient_of_variation"],
                "weak_scaling_efficiency": efficiency,
                "distributed_weak_scaling_efficiency": (
                    None
                    if distributed_baseline is None or world == 1
                    else (
                        {
                            "method": "distributed_baseline",
                            "baseline_world_size": 2,
                            "speedup": 1.0,
                            "confidence_level": 0.95,
                            "confidence_interval": [1.0, 1.0],
                        }
                        if world == 2
                        else {
                            **bootstrap_median_speedup(
                                distributed_baseline[phase]["samples_seconds"],
                                samples,
                            ),
                            "baseline_world_size": 2,
                        }
                    )
                ),
            }
        points.append(
            {
                "world_size": world,
                "n_wires": item["workload"]["n_wires"],
                "node_count": item["node_count"],
                "gate_count": item["workload"].get("gate_count"),
                "active_wire_count": item["workload"].get("active_wire_count"),
                "parameter_count": item["workload"].get("parameter_count"),
                "phases": phases,
                "forward_peak_memory_bytes": item["forward"]["peak_memory_bytes_max"],
                "end_to_end_peak_memory_bytes": item["end_to_end"][
                    "peak_memory_bytes_max"
                ],
                "backward_communication_bytes_per_rank_max": item[
                    "backward_communication_bytes_per_rank_max"
                ],
                "forward_communication_bytes_per_rank_max": item["forward"].get(
                    "communication_bytes_per_rank_max", 0
                ),
                "gradient_absolute_error_max": item["correctness"][
                    "gradient_absolute_error_max"
                ],
            }
        )
    return {
        "schema_version": SCHEMA,
        "benchmark": "statevector_differentiable_training_scaling_report",
        "artifact_class": "derived_development_report",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "definition": f"constant_2^{int(math.log2(local_amplitudes))}_amplitudes_per_rank",
        "display_efficiency_baseline_world_size": 1,
        "display_efficiency_rationale": (
            "the full weak-scaling series is normalized at the requested single-GPU "
            "baseline; distributed-baseline efficiency remains available per point"
        ),
        "baseline_n_wires": baseline_wires,
        "local_amplitudes": local_amplitudes,
        "workload": baseline["workload"],
        "points": points,
        "source_world_sizes": worlds,
        "validation_blockers": [
            "development_artifacts_not_release_scalability_evidence",
            "matched_external_simulator_comparison_pending",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts]
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
