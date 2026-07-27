"""Fail-closed profiling records for site-sharded MPS training."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

MPS_PROFILE_SCHEMA = "flagquantum.site_sharded_mps_profile.v1"
MPS_PROFILE_PHASES = (
    "forward",
    "reverse_vjp",
    "optimizer",
    "unattributed",
)

MPS_TRACE_OPERATION_LABELS = (
    "flagquantum::mps::forward",
    "flagquantum::mps::left_environment_scan",
    "flagquantum::mps::right_environment_scan",
    "flagquantum::mps::objective_adjoint",
    "flagquantum::mps::reverse_vjp",
    "flagquantum::mps::optimizer",
    "flagquantum::mps::two_site_split",
    "flagquantum::mps::qr",
)

MPS_TRACE_COMMUNICATION_LABELS = (
    "flagquantum::mps::p2p_send",
    "flagquantum::mps::p2p_recv",
    "flagquantum::mps::gradient_collective",
)


def workload_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return the stable identity used by all ISSUE-097 follow-up comparisons."""
    content = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def build_mps_critical_path_report(
    rank_records: Sequence[Mapping[str, Any]],
    *,
    workload_manifest: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
    warmup_steps: int,
) -> dict[str, Any]:
    """Aggregate rank-local step metrics without inventing measured timings."""
    if not rank_records:
        raise ValueError("MPS profile requires at least one rank record")
    ranks = tuple(sorted(int(record["rank"]) for record in rank_records))
    if ranks != tuple(range(len(rank_records))):
        raise ValueError("MPS profile requires contiguous unique rank records")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    samples: list[dict[str, Any]] = []
    message_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"message_count": 0, "payload_bytes": 0}
    )
    maximum_reconciliation_error = 0.0
    step_rank_seconds: dict[int, list[float]] = defaultdict(list)
    for record in rank_records:
        rank = int(record["rank"])
        steps = tuple(record.get("step_metrics", ()))
        for raw in steps:
            step = int(raw["step"])
            end_to_end = float(raw["end_to_end_seconds"])
            phases = {
                "forward": float(raw["forward_seconds"]),
                "reverse_vjp": float(raw["reverse_seconds"]),
                "optimizer": float(raw["optimizer_seconds"]),
            }
            phases["unattributed"] = max(0.0, end_to_end - sum(phases.values()))
            reconciled = sum(phases.values())
            error = abs(end_to_end - reconciled) / max(end_to_end, 1e-12)
            maximum_reconciliation_error = max(maximum_reconciliation_error, error)
            message_classes = {
                "boundary_forward_tensor": {
                    "message_count": int(raw.get("boundary_forward_exchanges", 0)) * 2,
                    "payload_bytes": int(raw.get("boundary_bytes", 0)) // 2,
                },
                "boundary_reverse_adjoint": {
                    "message_count": int(raw.get("boundary_reverse_exchanges", 0)) * 2,
                    "payload_bytes": int(raw.get("boundary_bytes", 0)) // 2,
                },
                "parameter_gradient_collective": {
                    "message_count": int(raw.get("gradient_collective_count", 0)),
                    "payload_bytes": int(raw.get("gradient_collective_bytes", 0)),
                },
            }
            for name, values in message_classes.items():
                message_totals[name]["message_count"] += values["message_count"]
                message_totals[name]["payload_bytes"] += values["payload_bytes"]
            step_rank_seconds[step].append(end_to_end)
            samples.append(
                {
                    "rank": rank,
                    "step": step,
                    "sample_class": "warm"
                    if step >= warmup_steps
                    else "cold_or_warmup",
                    "end_to_end_seconds": end_to_end,
                    "phase_seconds": phases,
                    "reconciliation_relative_error": error,
                    "message_classes": message_classes,
                    "svd_activity": int(raw.get("svd_activity", 0)),
                    "qr_activity": int(raw.get("qr_activity", 0)),
                }
            )
    imbalance = {
        str(step): {
            "minimum_seconds": min(values),
            "maximum_seconds": max(values),
            "idle_imbalance_seconds": max(values) - min(values),
            "max_over_min": max(values) / max(min(values), 1e-12),
        }
        for step, values in sorted(step_rank_seconds.items())
    }
    warm_samples = sum(item["sample_class"] == "warm" for item in samples)
    setup_by_rank = {
        str(int(record["rank"])): {
            "process_group_seconds": float(
                record.get("process_group_setup_seconds", 0.0)
            ),
            "communication_seconds": float(
                record.get("communication_setup_seconds", 0.0)
            ),
        }
        for record in rank_records
    }
    return {
        "schema": MPS_PROFILE_SCHEMA,
        "workload_manifest": dict(workload_manifest),
        "workload_sha256": workload_fingerprint(workload_manifest),
        "environment_manifest": dict(environment_manifest),
        "world_size": len(rank_records),
        "warmup_steps": warmup_steps,
        "warm_sample_count_across_ranks": warm_samples,
        "cold_setup_seconds_by_rank": setup_by_rank,
        "rank_step_samples": samples,
        "message_totals_by_payload_class": dict(message_totals),
        "rank_imbalance_by_step": imbalance,
        "maximum_reconciliation_relative_error": maximum_reconciliation_error,
        "phase_reconciliation_passed": maximum_reconciliation_error <= 0.05,
        "trace_contract": {
            "format": "chrome_trace",
            "required_record_function_prefix": "flagquantum::mps::",
            "required_operation_labels": MPS_TRACE_OPERATION_LABELS,
            "required_communication_labels": MPS_TRACE_COMMUNICATION_LABELS,
        },
        "performance_claim_allowed": False,
        "scalability_claim_allowed": False,
    }


__all__ = (
    "MPS_PROFILE_PHASES",
    "MPS_PROFILE_SCHEMA",
    "MPS_TRACE_COMMUNICATION_LABELS",
    "MPS_TRACE_OPERATION_LABELS",
    "build_mps_critical_path_report",
    "workload_fingerprint",
)
