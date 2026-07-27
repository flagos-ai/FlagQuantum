#!/usr/bin/env python3
"""Aggregate constant-local-state statevector weak-scaling measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from .statevector_scaling_report import bootstrap_median_speedup
except ImportError:  # pragma: no cover - direct script compatibility
    from statevector_scaling_report import bootstrap_median_speedup

SCHEMA = "flagquantum.statevector.weak_scaling_report.v1"
SOURCE_SCHEMA = "flagquantum.statevector.strong_scaling.v1"


def build_report(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one artifact required")
    by_world = {int(item.get("world_size", 0)): item for item in artifacts}
    if len(by_world) != len(artifacts) or 1 not in by_world:
        raise ValueError("unique power-of-two world sizes including one required")
    worlds = sorted(by_world)
    if any(world < 1 or world & (world - 1) for world in worlds):
        raise ValueError("weak-scaling world sizes must be powers of two")
    baseline = by_world[1]
    baseline_wires = int(baseline["workload"]["n_wires"])
    local_state_bytes: int | None = None
    for world, item in by_world.items():
        if item.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("invalid source schema")
        if item.get("artifact_class") != "measured_development_run":
            raise ValueError("all sources must be measured development runs")
        if item.get("backend") != "nccl" or not item.get("correctness", {}).get(
            "passed"
        ):
            raise ValueError("every source must pass NCCL correctness")
        workload = item["workload"]
        if int(workload["n_wires"]) != baseline_wires + int(math.log2(world)):
            raise ValueError("qubit count must grow by log2(world_size)")
        if workload["depth"] != baseline["workload"]["depth"]:
            raise ValueError("depth must remain fixed")
        rows = item.get("rank_timings", [])
        if len(rows) != world or len(item.get("rank_peak_memory_bytes", [])) != world:
            raise ValueError("incomplete rank evidence")
        state_sizes = {int(row["local_state_bytes"]) for row in rows}
        if len(state_sizes) != 1:
            raise ValueError("rank-local state sizes differ within a point")
        point_local_state = next(iter(state_sizes))
        if local_state_bytes is None:
            local_state_bytes = point_local_state
        elif point_local_state != local_state_bytes:
            raise ValueError("rank-local state size is not constant across points")
        if len(item.get("timing", {}).get("samples_seconds", [])) < 3:
            raise ValueError("insufficient timing samples")
        if int(item.get("node_count", 0)) > 1:
            tiers = item.get("communication_tiers", {})
            if tiers.get("route_classification") != "inter_node_collective":
                raise ValueError("multi-node point lacks inter-node evidence")

    baseline_samples = baseline["timing"]["samples_seconds"]
    points = []
    for world in worlds:
        item = by_world[world]
        samples = item["timing"]["samples_seconds"]
        ratio = (
            {
                "method": "baseline",
                "speedup": 1.0,
                "confidence_level": 0.95,
                "confidence_interval": [1.0, 1.0],
            }
            if world == 1
            else bootstrap_median_speedup(baseline_samples, samples)
        )
        points.append(
            {
                "world_size": world,
                "n_wires": int(item["workload"]["n_wires"]),
                "node_count": int(item["node_count"]),
                "median_seconds": statistics.median(samples),
                "weak_scaling_efficiency": ratio,
                "rank_peak_memory_bytes_max": max(item["rank_peak_memory_bytes"]),
                "local_state_bytes": local_state_bytes,
                "communication_route": item["communication_tiers"][
                    "route_classification"
                ],
                "coefficient_of_variation": item["timing"]["coefficient_of_variation"],
            }
        )
    return {
        "schema_version": SCHEMA,
        "benchmark": "statevector_weak_scaling_report",
        "artifact_class": "derived_development_report",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "definition": "constant_rank_local_amplitudes",
        "baseline_n_wires": baseline_wires,
        "depth": baseline["workload"]["depth"],
        "gate_count": baseline["workload"]["gate_count"],
        "dtype": baseline["workload"]["dtype"],
        "local_state_bytes": local_state_bytes,
        "points": points,
        "source_world_sizes": worlds,
        "validation_blockers": [
            "development_artifacts_not_release_scalability_evidence"
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
