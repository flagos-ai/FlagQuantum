"""Fail-closed validation for ISSUE-093 stability and recovery evidence."""

from __future__ import annotations

from typing import Any, Mapping


class MPSStabilityCertificationError(ValueError):
    pass


def _require_rank_timelines(soak: Mapping[str, Any]) -> None:
    steps = int(soak.get("steps", 0))
    warmup = int(soak.get("warmup_steps", -1))
    tolerance = int(soak.get("memory_growth_tolerance_bytes", -1))
    records = soak.get("rank_records", ())
    if len(records) != 8 or {int(item.get("rank", -1)) for item in records} != set(
        range(8)
    ):
        raise MPSStabilityCertificationError("eight per-rank soak records are required")
    fingerprints = set()
    generations = set()
    for record in records:
        timeline = record.get("memory_timeline", ())
        if len(timeline) != steps or [
            int(item.get("step", -1)) for item in timeline
        ] != list(range(steps)):
            raise MPSStabilityCertificationError(
                "per-rank memory timeline is incomplete"
            )
        if any(
            any(
                int(item.get(field, -1)) < 0
                for field in (
                    "allocated",
                    "reserved",
                    "peak",
                    "tape",
                    "optimizer",
                    "communication",
                )
            )
            for item in timeline
        ):
            raise MPSStabilityCertificationError(
                "memory timeline contains invalid counters"
            )
        stable = timeline[min(warmup, steps - 1) :]
        growth = max(0, int(stable[-1]["allocated"]) - int(stable[0]["allocated"]))
        if growth != int(record.get("memory_growth_bytes", -1)) or growth > tolerance:
            raise MPSStabilityCertificationError(
                "live-memory growth does not satisfy the declared bound"
            )
        if not record.get("rank_useful_work"):
            raise MPSStabilityCertificationError("a soak rank completed no useful work")
        if (
            float(record.get("restart_parameter_error", 1.0)) > 1e-7
            or float(record.get("restart_loss_error", 1.0)) > 1e-7
        ):
            raise MPSStabilityCertificationError("checkpoint numerical error exceeded")
        fingerprints.add(record.get("checkpoint_contract_fingerprint"))
        generations.add(int(record.get("restart_start_step", -1)))
    if (
        None in fingerprints
        or len(fingerprints) != 1
        or generations != {int(soak.get("restart_generation", -2))}
    ):
        raise MPSStabilityCertificationError(
            "checkpoint generation/contract agreement failed"
        )


def require_mps_stability(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "flagquantum.issue093.mps_stability_matrix.v2":
        raise MPSStabilityCertificationError("unexpected ISSUE-093 schema")
    soaks = payload.get("soaks", ())
    if len(soaks) != 2 or {item.get("optimizer") for item in soaks} != {"sgd", "adam"}:
        raise MPSStabilityCertificationError("SGD and Adam soak evidence is required")
    for soak in soaks:
        if int(soak.get("world_size", 0)) != 8 or int(soak.get("steps", 0)) < 100:
            raise MPSStabilityCertificationError("eight-rank 100-step soak is required")
        _require_rank_timelines(soak)
    capacity = payload.get("capacity_multistep", {})
    capacity_records = capacity.get("rank_records", ())
    if int(capacity.get("steps", 0)) < 3 or len(capacity_records) != 8:
        raise MPSStabilityCertificationError(
            "capacity multi-step evidence is incomplete"
        )
    if not all(len(item.get("memory_timeline", ())) >= 3 for item in capacity_records):
        raise MPSStabilityCertificationError("capacity memory timelines are incomplete")
    required_faults = {
        "topology_mismatch",
        "generation_mismatch",
        "interrupted_checkpoint",
        "rank_exception",
        "cuda_oom",
        "collective_timeout",
    }
    faults = payload.get("fault_matrix", ())
    if {item.get("fault") for item in faults} != required_faults:
        raise MPSStabilityCertificationError("fault matrix is incomplete")
    if not all(
        item.get("passed") is True
        and item.get("cleanup_verified") is True
        and int(item.get("exit_code", 0)) != 0
        and 0
        < float(item.get("elapsed_seconds", 0))
        <= float(item.get("timeout_seconds", 0))
        and len(str(item.get("log_sha256", ""))) == 64
        for item in faults
    ):
        raise MPSStabilityCertificationError(
            "fault termination/cleanup evidence is invalid"
        )
    sources = payload.get("source_artifacts", ())
    required_sources = {
        "adam_soak",
        "sgd_soak",
        "capacity_multistep",
        "fault_matrix",
        "workload",
    }
    if {item.get("kind") for item in sources} != required_sources or any(
        not item.get("path") or len(str(item.get("sha256", ""))) != 64
        for item in sources
    ):
        raise MPSStabilityCertificationError(
            "stability source provenance is incomplete"
        )
    if not payload.get("command") or not payload.get("commit"):
        raise MPSStabilityCertificationError("stability run provenance is incomplete")
    if payload.get("full_mps_materialization") is not False:
        raise MPSStabilityCertificationError("full MPS materialization is forbidden")
    if payload.get("scalability_claim_allowed") or payload.get("release_gate_allowed"):
        raise MPSStabilityCertificationError(
            "stability evidence cannot promote release"
        )


__all__ = ("MPSStabilityCertificationError", "require_mps_stability")
