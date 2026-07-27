"""Fail-closed readiness gate for ISSUE-044 production acceptance."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from flagquantum.runtime.observability.evidence import verify_evidence_artifact

MANIFEST = Path("benchmarks/manifests/issue044_statevector_release_v2.json")
RESULTS = Path("benchmarks/results/scalability")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "flagquantum.issue044.release_manifest.v1",
        "flagquantum.issue044.release_manifest.v2",
    }:
        raise ValueError("invalid ISSUE-044 manifest schema")
    if not payload.get("frozen_before_release_run"):
        raise ValueError("ISSUE-044 thresholds must be frozen before measurement")
    return payload


def evaluate_issue044_release(
    artifacts: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    signing_key: bytes | None = None,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    production: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    rejected_integrity = False
    for artifact in artifacts:
        if artifact.get("artifact_class") != "measured_production_run":
            continue
        evidence = artifact.get("evidence")
        provenance = artifact.get("provenance")
        if not isinstance(evidence, Mapping) or not isinstance(provenance, Mapping):
            rejected_integrity = True
            continue
        if signing_key is not None:
            valid, _ = verify_evidence_artifact(artifact, signing_key=signing_key)
            if not valid:
                rejected_integrity = True
                continue
        if evidence.get("release_gate_allowed") is True:
            production.append((evidence, provenance))
    if rejected_integrity:
        blockers.append("invalid_or_unsigned_production_artifact")

    worlds = {_as_int(evidence.get("world_size")) for evidence, _ in production}
    required_worlds = set(manifest["topologies"]["single_node_world_sizes"])
    if not required_worlds <= worlds:
        blockers.append("missing_release_world_sizes")
    if not any(
        _as_int(evidence.get("node_count")) >= 2 for evidence, _ in production
    ):
        blockers.append("missing_multinode_correctness_artifact")

    speed = manifest["speed_workload"]
    speed_candidates = [
        (evidence, provenance)
        for evidence, provenance in production
        if evidence.get("acceptance_case") == "matched_speed"
    ]
    speed_accepted = any(
        _as_float(evidence.get("speedup")) >= _as_float(speed["minimum_speedup"])
        and _confidence_interval_lower(evidence) > _as_float(
            speed["confidence_interval_must_exclude_speedup"]
        )
        and _as_int(provenance.get("warmup")) >= _as_int(speed["warmup_steps"])
        and _as_int(provenance.get("iterations"))
        >= _as_int(speed["measured_steps"])
        and _as_float(evidence.get("scaling_efficiency"))
        >= _as_float(speed["minimum_scaling_efficiency"])
        for evidence, provenance in speed_candidates
    )
    if not speed_accepted:
        blockers.append("missing_statistically_significant_speedup_artifact")

    capacity = manifest["capacity_workload"]
    single_gpu_oom = any(
        evidence.get("acceptance_case") == "single_gpu_capacity_failure"
        and _as_int(evidence.get("world_size")) == 1
        and evidence.get("single_device_oom_observed") is True
        and evidence.get("capacity_failure_reason")
        and evidence.get("workload_sha256") == capacity["workload_sha256"]
        for evidence, _ in production
    )
    if not single_gpu_oom:
        blockers.append("missing_single_gpu_measured_oom_artifact")
    multi_gpu_completion = any(
        evidence.get("acceptance_case") == "multi_gpu_capacity_completion"
        and _as_int(evidence.get("world_size")) > 1
        and _as_int(evidence.get("training_step_count"))
        >= _as_int(capacity["minimum_optimizer_steps"])
        and evidence.get("full_state_materialized") is False
        and evidence.get("workload_sha256") == capacity["workload_sha256"]
        for evidence, _ in production
    )
    if not multi_gpu_completion:
        blockers.append("missing_multi_gpu_capacity_completion_artifact")

    required_fields = set(manifest["required_artifact_fields"])
    if production and any(
        not _has_required_fields(evidence, provenance, required_fields)
        for evidence, provenance in production
    ):
        blockers.append("production_artifact_missing_required_fields")
    return not blockers, tuple(blockers)


def _confidence_interval_lower(evidence: Mapping[str, Any]) -> float:
    interval = evidence.get("speedup_confidence_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return float("-inf")
    return _as_float(interval[0])


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _has_required_fields(
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    required_fields: set[str],
) -> bool:
    provenance_fields = {
        "raw_log_sha256",
        "commit",
        "workload_sha256",
        "rank_mapping",
    }
    aliases = {
        "hardware_inventory": "devices",
        "rank_mapping": "rank_mapping",
    }
    for field in required_fields:
        if field in provenance_fields:
            value = provenance.get(field)
        elif field == "hardware_inventory":
            value = evidence.get(field, provenance.get(aliases[field]))
        else:
            value = evidence.get(field)
        if value in (None, "", (), []):
            return False
    return True


def main() -> None:
    manifest = load_manifest()
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS.glob("*.json"))
    ]
    signing_key = os.environ.get("FQ_EVIDENCE_SIGNING_KEY", "").encode()
    passed, blockers = evaluate_issue044_release(
        artifacts, manifest, signing_key=signing_key
    )
    print(json.dumps({"issue": "ISSUE-044", "passed": passed, "blockers": blockers}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
