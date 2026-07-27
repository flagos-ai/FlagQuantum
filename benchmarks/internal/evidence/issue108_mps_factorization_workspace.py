#!/usr/bin/env python3
"""Audit ISSUE-108 workspace-aware factorization evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "flagquantum.issue108.mps_factorization_workspace.v1"


def audit_factorization_workspace(
    payload: dict[str, Any],
    *,
    expected_world_size: int = 8,
    required_max_bond: int = 1024,
    minimum_saturation_depth: int = 7,
) -> dict[str, Any]:
    """Fail closed on missing headroom, budget, allocator, or chunk evidence."""

    blockers: list[str] = []
    workload = payload.get("workload", {})
    environment = payload.get("environment", {})
    ranks: Sequence[dict[str, Any]] = payload.get("rank_records", ())
    if payload.get("status") != "passed" or not payload.get("audit", {}).get("passed"):
        blockers.append("forward_plateau_artifact_not_passed")
    if int(workload.get("max_bond", 0)) != required_max_bond:
        blockers.append("max_bond_mismatch")
    if max(workload.get("depths", (0,))) < minimum_saturation_depth:
        blockers.append("saturation_depth_missing")
    if int(environment.get("world_size", 0)) != expected_world_size:
        blockers.append("world_size_mismatch")
    if len(ranks) != expected_world_size:
        blockers.append("rank_record_count_mismatch")

    rank_audits = []
    seen_ranks: set[int] = set()
    for rank_record in ranks:
        rank = int(rank_record.get("rank", -1))
        if rank in seen_ranks:
            blockers.append(f"rank_{rank}_duplicate")
        seen_ranks.add(rank)
        summary = rank_record.get("factorization_summary", {})
        records = rank_record.get("factorization_records", ())
        if not records or int(summary.get("decision_count", -1)) != len(records):
            blockers.append(f"rank_{rank}_factorization_records_missing")
        if int(summary.get("allocator_retry_count", -1)) != 0:
            blockers.append(f"rank_{rank}_allocator_retry")
        if int(summary.get("allocator_oom_count", -1)) != 0:
            blockers.append(f"rank_{rank}_allocator_oom")
        minimum_free = int(summary.get("minimum_observed_free_bytes", 0))
        minimum_headroom = int(summary.get("minimum_headroom_bytes", 0))
        if minimum_headroom < 2 * (1 << 30):
            blockers.append(f"rank_{rank}_headroom_policy_too_small")
        if minimum_free < minimum_headroom:
            blockers.append(f"rank_{rank}_observed_headroom_breached")
        high_bond_records = []
        for index, record in enumerate(records):
            selected = int(record.get("selected_chunk_size", 0))
            requested = int(record.get("requested_chunk_size", 0))
            available = int(record.get("available_working_bytes", -1))
            selected_bytes = int(record.get("selected_working_set_bytes", 0))
            if selected < 1 or selected > requested:
                blockers.append(f"rank_{rank}_decision_{index}_invalid_chunk")
            if selected_bytes > available:
                blockers.append(f"rank_{rank}_decision_{index}_budget_exceeded")
            shape = record.get("shape", ())
            dimensions = [int(value) for item in shape for value in item]
            if dimensions and max(dimensions) >= required_max_bond:
                high_bond_records.append(record)
                if selected > 2:
                    blockers.append(
                        f"rank_{rank}_decision_{index}_high_bond_chunk_exceeded"
                    )
        if not high_bond_records:
            blockers.append(f"rank_{rank}_high_bond_evidence_missing")
        rank_audits.append(
            {
                "rank": rank,
                "decision_count": len(records),
                "high_bond_decision_count": len(high_bond_records),
                "minimum_observed_free_bytes": minimum_free,
                "minimum_headroom_bytes": minimum_headroom,
                "maximum_selected_working_set_bytes": int(
                    summary.get("maximum_selected_working_set_bytes", 0)
                ),
                "selected_chunk_histogram": summary.get(
                    "selected_chunk_histogram", {}
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "passed": not blockers,
        "expected_world_size": expected_world_size,
        "required_max_bond": required_max_bond,
        "minimum_saturation_depth": minimum_saturation_depth,
        "rank_audits": rank_audits,
        "blockers": tuple(sorted(set(blockers))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    audit = audit_factorization_workspace(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"ISSUE-108 workspace audit failed: {audit['blockers']}")


if __name__ == "__main__":
    main()
