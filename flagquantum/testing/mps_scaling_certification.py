"""Fail-closed statistical scaling certification for production MPS evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


class MPSScalingCertificationError(ValueError):
    """Raised when an ISSUE-095 artifact is incomplete or post-selected."""


def _fail(message: str) -> None:
    raise MPSScalingCertificationError(message)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def _close(actual: Any, expected: float) -> bool:
    return math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12)


def require_mps_scaling(
    payload: Mapping[str, Any], *, manifest_root: Path | None = None
) -> Mapping[str, Any]:
    """Validate raw-sample completeness, predeclaration, regions, and promotion."""
    if payload.get("schema") != "flagquantum.issue095.mps_scaling.v1":
        _fail("unexpected ISSUE-095 schema")
    pre = payload.get("predeclaration", {})
    if not pre.get("frozen_before_measurement"):
        _fail("workloads were not frozen before measurement")
    manifest_path = pre.get("path")
    digest = pre.get("sha256")
    if not manifest_path or not digest:
        _fail("predeclaration path and digest are required")
    if manifest_root is not None:
        path = manifest_root / manifest_path
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            _fail("predeclaration manifest digest mismatch")

    runs = payload.get("runs", [])
    expected = {
        (family, mode, world)
        for family in (
            "tfim_time_evolution",
            "circuit_training",
            "variable_bond_training",
        )
        for mode in ("strong", "weak")
        for world in (1, 2, 4, 8)
    }
    found: set[tuple[str, str, int]] = set()
    environment = payload.get("environment", {})
    immutable = environment.get("snapshot_sha256")
    if not all(
        (
            immutable,
            environment.get("source_commit"),
            environment.get("environment_sha256"),
        )
    ):
        _fail("immutable source/environment snapshot is required")
    for run in runs:
        key = (run.get("family"), run.get("scaling_mode"), run.get("world_size"))
        if key in found:
            _fail(f"duplicate/post-selected run: {key}")
        found.add(key)
        samples = run.get("raw_samples_seconds", [])
        if run.get("warmup") != 5 or len(samples) < 20:
            _fail(f"incomplete fixed sample set: {key}")
        if run.get("sample_count") != len(samples):
            _fail(f"sample-count mismatch: {key}")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in samples
        ):
            _fail(f"invalid timing sample: {key}")
        if run.get("snapshot_sha256") != immutable:
            _fail(f"mixed source/environment snapshots: {key}")
        for field in (
            "mean_seconds",
            "p50_seconds",
            "p95_seconds",
            "coefficient_of_variation",
            "confidence_interval_95_seconds",
            "peak_memory_bytes",
        ):
            if field not in run:
                _fail(f"missing {field}: {key}")
        numeric = [float(value) for value in samples]
        mean = statistics.mean(numeric)
        deviation = statistics.stdev(numeric)
        margin = 2.093 * deviation / math.sqrt(len(numeric))
        expected_stats = {
            "mean_seconds": mean,
            "p50_seconds": _percentile(numeric, 0.50),
            "p95_seconds": _percentile(numeric, 0.95),
            "coefficient_of_variation": deviation / mean,
        }
        if any(not _close(run[name], value) for name, value in expected_stats.items()):
            _fail(f"statistics do not match retained samples: {key}")
        interval = run["confidence_interval_95_seconds"]
        if len(interval) != 2 or not all(
            _close(actual, expected)
            for actual, expected in zip(interval, (mean - margin, mean + margin))
        ):
            _fail(f"confidence interval does not match retained samples: {key}")
        components = run.get("component_seconds", {})
        required_components = {
            "initialization",
            "forward",
            "reverse",
            "svd_qr",
            "optimizer",
            "communication",
            "checkpoint",
            "end_to_end",
        }
        if set(components) != required_components:
            _fail(f"incomplete component attribution: {key}")
        if any(
            value is None or not math.isfinite(float(value))
            for value in components.values()
        ):
            _fail(f"unmeasured component attribution: {key}")
        if (
            run.get("rank_imbalance") is None
            or run.get("communication_fraction") is None
        ):
            _fail(f"rank/communication measurements are missing: {key}")
    if found != expected:
        _fail(f"incomplete predeclared matrix; missing={sorted(expected - found)}")

    regions = payload.get("regions", {})
    if set(regions) != {
        "local_faster",
        "distributed_speedup",
        "weak_scaling",
        "capacity_only",
    }:
        _fail("four disjoint decision regions are required")
    if any(not regions[name] for name in regions):
        _fail("each crossover decision region requires measured evidence")
    members = [
        json.dumps(item, sort_keys=True)
        for values in regions.values()
        for item in values
    ]
    if len(members) != len(set(members)):
        _fail("crossover regions overlap")
    gate = payload.get("performance_gate", {})
    if gate.get("passed") and not any(
        item.get("speedup_ci_low", 0.0) > 1.0 for item in regions["distributed_speedup"]
    ):
        _fail("performance promotion lacks a speedup CI lower bound above 1")
    audit = payload.get("audit", {})
    if not audit.get("complete_predeclared_matrix"):
        _fail("benchmark audit did not approve the complete matrix")
    artifacts = audit.get("source_artifacts", ())
    if manifest_root is not None:
        if not artifacts:
            _fail("source artifact hashes are required")
        for artifact in artifacts:
            source = Path(artifact.get("path", ""))
            source = source if source.is_absolute() else manifest_root / source
            if not source.is_file() or hashlib.sha256(
                source.read_bytes()
            ).hexdigest() != artifact.get("sha256"):
                _fail(f"source artifact digest mismatch: {artifact.get('path')}")
    if not payload.get("repeatability", {}).get("within_declared_uncertainty"):
        _fail("repeated sample partitions did not reproduce the conclusion")
    if payload.get("release_gate_allowed"):
        _fail("ISSUE-095 crossover evidence cannot independently approve release")
    return payload


__all__ = ["MPSScalingCertificationError", "require_mps_scaling"]
