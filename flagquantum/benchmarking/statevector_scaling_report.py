#!/usr/bin/env python3
"""Aggregate compatible strong-scaling artifacts with bootstrap confidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "flagquantum.statevector.scaling_report.v1"
SOURCE_SCHEMA = "flagquantum.statevector.strong_scaling.v1"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_median_speedup(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 440044,
) -> dict[str, Any]:
    if len(baseline) < 3 or len(candidate) < 3 or resamples < 1000:
        raise ValueError("at least three samples and 1000 resamples required")
    generator = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        base = [generator.choice(baseline) for _ in baseline]
        target = [generator.choice(candidate) for _ in candidate]
        ratios.append(statistics.median(base) / statistics.median(target))
    point = statistics.median(baseline) / statistics.median(candidate)
    return {
        "method": "independent_bootstrap_median_ratio",
        "seed": seed,
        "resamples": resamples,
        "speedup": point,
        "confidence_level": 0.95,
        "confidence_interval": [
            _percentile(ratios, 0.025),
            _percentile(ratios, 0.975),
        ],
    }


def build_report(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one artifact required")
    hashes = {item.get("workload_sha256") for item in artifacts}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("all artifacts must have the same workload_sha256")
    worlds = [int(item.get("world_size", 0)) for item in artifacts]
    if len(set(worlds)) != len(worlds) or 1 not in worlds:
        raise ValueError("artifacts require unique world sizes including world_size=1")
    if any(not item.get("correctness", {}).get("passed") for item in artifacts):
        raise ValueError("all source artifacts must pass correctness")
    for item, world_size in zip(artifacts, worlds):
        if item.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("all source artifacts require strong-scaling schema")
        if item.get("artifact_class") != "measured_development_run":
            raise ValueError("all source artifacts must be measured development runs")
        if item.get("backend") != "nccl":
            raise ValueError("all source artifacts require NCCL")
        if item.get("scalability_claim_allowed") is not False:
            raise ValueError("source artifacts must reject scalability claims")
        if item.get("release_gate_allowed") is not False:
            raise ValueError("source artifacts must reject release claims")
        if len(item.get("rank_timings", [])) != world_size:
            raise ValueError("source artifact has incomplete rank timings")
        if len(item.get("rank_peak_memory_bytes", [])) != world_size:
            raise ValueError("source artifact has incomplete rank memory")
        if (
            len(item.get("hardware_inventory", {}).get("cuda_device_names", []))
            != world_size
        ):
            raise ValueError("source artifact has incomplete hardware inventory")
        if len(item.get("timing", {}).get("samples_seconds", [])) < 3:
            raise ValueError("source artifact has insufficient timing samples")
        node_count = int(item.get("node_count", 1))
        if node_count > 1:
            placements = item.get("rank_placement", [])
            if (
                len(placements) != world_size
                or len({row.get("hostname") for row in placements}) != node_count
            ):
                raise ValueError("multi-node source has incomplete rank placement")
            tiers = item.get("communication_tiers", {})
            if not (
                tiers.get("route_classification") == "inter_node_collective"
                and int(tiers.get("inter_node_logical_bytes_per_rank_max", 0)) > 0
            ):
                raise ValueError("multi-node source lacks inter-node traffic evidence")
    by_world = {int(item["world_size"]): item for item in artifacts}
    reference_invariants = by_world[1]["correctness"]["global_invariants"]
    tolerance = max(
        float(item["correctness"].get("absolute_tolerance", 0.0)) for item in artifacts
    )
    for item in artifacts:
        invariants = item["correctness"].get("global_invariants", {})
        if (
            abs(
                float(invariants.get("norm", float("inf")))
                - float(reference_invariants["norm"])
            )
            > tolerance
        ):
            raise ValueError("global norm differs across world sizes")
        if set(invariants.get("z_expectations", {})) != set(
            reference_invariants["z_expectations"]
        ):
            raise ValueError("observable keys differ across world sizes")
        for wire, expected in reference_invariants["z_expectations"].items():
            if (
                abs(float(invariants["z_expectations"][wire]) - float(expected))
                > tolerance
            ):
                raise ValueError("global observables differ across world sizes")
    baseline = by_world[1]["timing"]["samples_seconds"]
    points = []
    for world_size in sorted(by_world):
        item = by_world[world_size]
        samples = item["timing"]["samples_seconds"]
        estimate = (
            {
                "method": "baseline",
                "speedup": 1.0,
                "confidence_level": 0.95,
                "confidence_interval": [1.0, 1.0],
            }
            if world_size == 1
            else bootstrap_median_speedup(baseline, samples)
        )
        points.append(
            {
                "world_size": world_size,
                "local_world_size": int(item.get("local_world_size", world_size)),
                "node_count": int(item.get("node_count", 1)),
                "median_seconds": statistics.median(samples),
                "speedup": estimate,
                "scaling_efficiency": estimate["speedup"] / world_size,
                "rank_peak_memory_bytes_max": max(item["rank_peak_memory_bytes"]),
                "communication_fraction": item["communication_fraction"],
                "communication_route": item.get("communication_tiers", {}).get(
                    "route_classification", "not_recorded_in_legacy_source"
                ),
                "source_artifact_class": item["artifact_class"],
            }
        )
    return {
        "schema_version": SCHEMA,
        "benchmark": "statevector_strong_scaling_report",
        "artifact_class": "derived_development_report",
        "claim_evidence_type": "accelerator_development_performance",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "workload_sha256": next(iter(hashes)),
        "workload": by_world[1]["workload"],
        "points": points,
        "source_world_sizes": sorted(worlds),
        "validation_blockers": [
            "development_artifacts_not_release_scalability_evidence"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    artifacts = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts
    ]
    report = build_report(artifacts)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
