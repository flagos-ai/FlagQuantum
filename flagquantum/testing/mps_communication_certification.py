"""Fail-closed ISSUE-094 NCCL communication evidence validation."""

from __future__ import annotations

from typing import Any, Mapping


class MPSCommunicationCertificationError(ValueError):
    pass


def require_mps_communication(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "flagquantum.issue094.mps_communication_matrix.v1":
        raise MPSCommunicationCertificationError("unexpected ISSUE-094 schema")
    if payload.get("protocol") != "batched_isend_irecv_descriptor_payload":
        raise MPSCommunicationCertificationError("unbatched MPS transport")
    if not payload.get("dedicated_cuda_stream"):
        raise MPSCommunicationCertificationError(
            "dedicated communication stream missing"
        )
    runs = payload.get("runs", ())
    if {run.get("world_size") for run in runs} != {2, 4, 8}:
        raise MPSCommunicationCertificationError(
            "2/4/8-rank communication runs required"
        )
    for run in runs:
        world = int(run["world_size"])
        if (
            run.get("setup_seconds_max", 0) <= 0
            or run.get("steady_step_seconds_mean", 0) <= 0
        ):
            raise MPSCommunicationCertificationError("setup/steady timing is missing")
        if run.get("unbatched_warning_count") != 0:
            raise MPSCommunicationCertificationError("steady path emitted P2P warning")
        trace = run.get("boundary_trace", ())
        if len(trace) != world - 1:
            raise MPSCommunicationCertificationError("boundary trace is incomplete")
        expected = [(rank, rank + 1) for rank in range(world - 1)]
        if [tuple(item["owner_ranks"]) for item in trace] != expected:
            raise MPSCommunicationCertificationError("boundary peers differ from plan")
        if [item["communication_sequence"] for item in trace] != list(
            range(world, 2 * world - 1)
        ):
            raise MPSCommunicationCertificationError(
                "communication sequence differs from deterministic plan"
            )
        if not all(
            item["forward_transport"] == "batched_isend_irecv"
            and item["reverse_transport"] == "batched_isend_irecv"
            for item in trace
        ):
            raise MPSCommunicationCertificationError("trace contains legacy transport")
        detailed = run.get("trace_by_rank_step")
        if detailed is not None:
            required = {
                "rank",
                "step",
                "message_count",
                "payload_bytes",
                "route",
                "synchronization_seconds",
                "straggler_wait_seconds",
                "updates",
            }
            if not detailed or any(required - set(item) for item in detailed):
                raise MPSCommunicationCertificationError(
                    "per-rank per-step communication report is incomplete"
                )
            if any(
                item["message_count"] < 0
                or item["payload_bytes"] < 0
                or item["synchronization_seconds"] < 0
                or item["straggler_wait_seconds"] < 0
                for item in detailed
            ):
                raise MPSCommunicationCertificationError(
                    "communication report contains invalid measurements"
                )
    baseline = payload.get("legacy_baseline", {})
    if not baseline.get("warning_observed") or baseline.get("mean_seconds", 0) <= 0:
        raise MPSCommunicationCertificationError("legacy comparison is missing")
    if payload.get("issue091_max_gradient_error", 1) > 3e-4:
        raise MPSCommunicationCertificationError("batched transport broke gradients")
    fault = payload.get("sequence_mismatch_fault", {})
    if not fault.get("passed") or not fault.get("bounded_cleanup"):
        raise MPSCommunicationCertificationError("sequence fault did not fail closed")
    if payload.get("full_mps_materialization") is not False:
        raise MPSCommunicationCertificationError(
            "full MPS materialization is forbidden"
        )
    if payload.get("scalability_claim_allowed") or payload.get("release_gate_allowed"):
        raise MPSCommunicationCertificationError(
            "communication evidence cannot promote release"
        )


__all__ = ("MPSCommunicationCertificationError", "require_mps_communication")
