"""Fail-closed ISSUE-096 multi-node and accelerator portability certification."""

from __future__ import annotations

import math
import string
from typing import Any, Mapping


class MPSPortabilityCertificationError(ValueError):
    """Raised when portability evidence does not prove the production envelope."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MPSPortabilityCertificationError(message)


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in string.hexdigits for character in text)


def require_mps_portability(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require measured two-node execution, portability, faults, and cleanup."""
    _require(
        payload.get("schema") == "flagquantum.issue096.mps_portability.v1",
        "unexpected ISSUE-096 schema",
    )
    _require(
        payload.get("evidence_source") == "measured_runtime",
        "measured runtime evidence required",
    )
    _require(
        int(payload.get("node_count", 0)) >= 2,
        "at least two physical nodes are required",
    )
    placements = payload.get("rank_placements", [])
    _require(
        len(placements) == payload.get("world_size"),
        "complete rank placement is required",
    )
    world = int(payload.get("world_size", 0))
    _require(
        sorted(item.get("rank") for item in placements) == list(range(world)),
        "rank placement must contain every unique rank",
    )
    _require(
        len({item.get("hostname") for item in placements}) >= 2,
        "rank placement must measure two hostnames",
    )
    _require(
        len({item.get("hostname") for item in placements}) == payload.get("node_count"),
        "node count differs from measured host placement",
    )
    _require(
        all(item.get("device_uuid") for item in placements),
        "measured device UUIDs are required",
    )
    environment = payload.get("environment", {})
    _require(
        environment.get("fabric") not in (None, "", "unspecified"),
        "measured production fabric is required",
    )
    snapshot = payload.get("source_snapshot", {})
    _require(
        all(
            snapshot.get(name)
            for name in ("source_commit", "environment_sha256", "snapshot_sha256")
        ),
        "runtime source/environment snapshot is required",
    )
    evidence = payload.get("evidence_artifacts", {})
    for name, schema in (
        ("parity", "flagquantum.issue096.parity.v1"),
        ("fault_lifecycle", "flagquantum.issue096.fault_lifecycle.v1"),
    ):
        item = evidence.get(name, {})
        _require(
            item.get("schema") == schema
            and item.get("snapshot_sha256") == snapshot["snapshot_sha256"]
            and _sha256(item.get("artifact_sha256")),
            f"{name} evidence provenance is invalid",
        )
    communication = payload.get("communication_tiers", {})
    _require(communication.get("measured"), "communication tiers must be measured")
    _require(
        communication.get("topology_aware_boundary_placement"),
        "topology-aware boundary placement is required",
    )
    _require(
        int(communication.get("intra_node_bytes", -1)) >= 0,
        "intra-node bytes are required",
    )
    _require(
        int(communication.get("inter_node_bytes", 0)) > 0,
        "executed inter-node bytes are required",
    )
    training = payload.get("training", {})
    _require(training.get("variable_bond"), "variable-bond workload is required")
    _require(
        training.get("forward")
        and training.get("backward")
        and training.get("optimizer"),
        "forward/backward/optimizer must execute",
    )
    _require(
        training.get("full_mps_materialization") is False,
        "full-MPS materialization is forbidden",
    )
    _require(
        training.get("numerical_parity_passed"),
        "ISSUE-091 numerical parity is required",
    )
    _require(
        training.get("capacity_parity_passed"), "ISSUE-092 capacity parity is required"
    )
    _require(
        training.get("stability_parity_passed"),
        "ISSUE-093 stability parity is required",
    )
    _require(
        training.get("checkpoint_restart_completed"),
        "measured checkpoint restart is required",
    )
    parity = training.get("parity", {})
    for name in ("value", "gradient", "parameter"):
        error = parity.get(f"max_{name}_error")
        tolerance = parity.get(f"{name}_tolerance")
        _require(
            error is not None
            and tolerance is not None
            and math.isfinite(float(error))
            and float(error) <= float(tolerance),
            f"{name} parity measurement exceeds tolerance",
        )
    lifecycle = payload.get("lifecycle", {})
    for field in (
        "rendezvous",
        "network_reachability",
        "timeout",
        "teardown",
        "checkpoint_restart",
        "one_rank_failure",
    ):
        _require(
            lifecycle.get(field, {}).get("passed"), f"lifecycle check failed: {field}"
        )
    _require(
        lifecycle.get("one_rank_failure", {}).get("partial_artifact_promoted") is False,
        "rank failure promoted a partial artifact",
    )
    _require(
        payload.get("cleanup", {}).get("orphan_process_count") == 0,
        "orphan processes remain",
    )
    matrix = payload.get("support_matrix", {})
    _require(
        all(name in matrix for name in ("tested", "experimental", "unsupported")),
        "support matrix must separate tested, experimental, and unsupported",
    )
    tested_classes = {
        (item.get("accelerator_class"), item.get("topology"))
        for item in matrix["tested"]
    }
    _require(
        len(tested_classes) >= 2,
        "a second accelerator class or PCIe topology is required",
    )
    _require(
        all(
            item.get("evidence_source") == "measured_runtime"
            and _sha256(item.get("artifact_sha256"))
            and item.get("accelerator_class")
            and item.get("topology")
            for item in matrix["tested"]
        ),
        "tested hardware classes require measured artifact provenance",
    )
    _require(not payload.get("blockers"), "portability certification has blockers")
    _require(
        payload.get("release_gate_allowed") is True,
        "release gate must be explicitly approved",
    )
    return payload


__all__ = ("MPSPortabilityCertificationError", "require_mps_portability")
