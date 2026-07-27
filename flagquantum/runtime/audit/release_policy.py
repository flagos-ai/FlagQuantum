"""Audit helpers for distributed scalability semantics.

These helpers enforce the product rule that multi-rank execution is only a
scalability result when one logical workload is actually sharded across ranks.
They are intentionally lightweight so runtime summaries, benchmark payloads,
and tests can share the same checks.
"""
# ruff: noqa: F401

from __future__ import annotations

from typing import Any, Mapping

try:
    from .vocabulary import (
        CLAIM_EVIDENCE_TYPES,
        CLAIMABILITY_STATUSES,
        EVIDENCE_BACKEND_FAMILIES,
        INCOMPLETE_DISTRIBUTION_SEMANTICS,
        MPS_BACKWARD_READINESS_STATUSES,
        MPS_STATE_MODES,
        MULTI_NODE_TRANSPORT_BACKENDS,
        RELEASE_CLAIM_EVIDENCE_TYPES,
        REPLICATED_DISTRIBUTION_SEMANTICS,
        SCALABLE_DISTRIBUTION_SEMANTICS,
        SINGLE_DEVICE_DISTRIBUTION_SEMANTICS,
        STATEVECTOR_STATE_MODES,
        STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES,
        TRANSPORT_EVIDENCE_STATUSES,
    )
except ImportError:  # legacy benchmark file-loader path
    from flagquantum.runtime.audit.vocabulary import (  # type: ignore
        CLAIM_EVIDENCE_TYPES,
        INCOMPLETE_DISTRIBUTION_SEMANTICS,
        RELEASE_CLAIM_EVIDENCE_TYPES,
        REPLICATED_DISTRIBUTION_SEMANTICS,
        SCALABLE_DISTRIBUTION_SEMANTICS,
        SINGLE_DEVICE_DISTRIBUTION_SEMANTICS,
    )


from .engine import (
    _backend_family,
    _communication_evidence_reported,
    _has_nonempty,
    _is_nonempty,
    _ownership_sharded,
    _payload_blockers,
    _rank_memory_reported,
    _release_transport_evidence_errors,
    _semantics,
    _training_step_count,
)
from .errors import DistributedScalabilityError
from .mps_readiness import (
    _mps_communication_plan_reported,
    _mps_evidence_reported,
    _mps_memory_plan_reported,
)
from .schema import (
    DistributedScalabilityAudit,
)


def _claim_evidence_type(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("claim_evidence_type", payload.get("evidence_type"))
    if explicit is not None:
        evidence_type = str(explicit)
        return evidence_type if evidence_type in CLAIM_EVIDENCE_TYPES else "unknown"

    if payload.get("release_payload") or payload.get("release_gate_payload"):
        return "release_payload"
    if payload.get("sharding_plan_available") or payload.get("planner"):
        return "plan_preflight"

    policy = (
        payload.get("distributed_backend_policy") or payload.get("backend_policy") or {}
    )
    policy_profile = ""
    if isinstance(policy, Mapping):
        policy_profile = str(policy.get("profile", ""))
    blockers = tuple(
        str(item) for item in payload.get("scalability_blockers", ()) or ()
    )
    executor = str(payload.get("executor", ""))
    if (
        policy_profile == "development"
        or "development" in executor
        or "single_process_development_simulator" in blockers
    ):
        return "development_smoke"

    if (
        policy_profile == "production"
        or str(payload.get("distributed_profile", "")) == "production"
    ):
        return "production_runtime"
    return "unknown"


def audit_distributed_scalability(
    payload: Mapping[str, Any],
) -> DistributedScalabilityAudit:
    """Audit whether a payload can honestly claim distributed scalability.

    Parameters
    ----------
    payload:
        A runtime ``summary()`` dictionary or benchmark JSON payload.
    """

    semantics = str(payload.get("distribution_semantics", "unknown"))
    claim = bool(payload.get("scalability_claim_allowed", False))
    evidence_type = _claim_evidence_type(payload)
    world_size = int(payload.get("world_size", 1) or 1)
    errors: list[str] = []
    warnings: list[str] = []

    if semantics == "unknown":
        errors.append("missing distribution_semantics")

    if claim and semantics not in SCALABLE_DISTRIBUTION_SEMANTICS:
        errors.append(
            "scalability_claim_allowed requires distribution_semantics='sharded_across_ranks'"
        )

    if semantics in SINGLE_DEVICE_DISTRIBUTION_SEMANTICS and claim:
        errors.append("single-device fast paths cannot claim distributed scalability")

    if semantics in SINGLE_DEVICE_DISTRIBUTION_SEMANTICS and world_size > 1:
        errors.append("single-device fast paths require world_size <= 1")

    if semantics in REPLICATED_DISTRIBUTION_SEMANTICS and claim:
        errors.append("replicated execution cannot claim single-workload scalability")

    if semantics in INCOMPLETE_DISTRIBUTION_SEMANTICS and claim:
        errors.append("incomplete or hybrid execution cannot claim full scalability")

    if semantics in SCALABLE_DISTRIBUTION_SEMANTICS:
        if world_size <= 1:
            errors.append("sharded_across_ranks requires world_size > 1")
        if "local_world_size" not in payload:
            errors.append("sharded payload must report local_world_size")
        if "node_count" not in payload:
            errors.append("sharded payload must report node_count")
        if not _has_nonempty(
            payload,
            ("rank_placement", "rank_shards", "shards", "local_memory_bytes_by_rank"),
        ):
            errors.append(
                "sharded payload must report per-rank ownership or rank placement"
            )
        if not _rank_memory_reported(payload):
            errors.append("sharded payload must report per-rank memory evidence")
        if not _communication_evidence_reported(payload):
            errors.append("sharded payload must report communication evidence")
        if not claim:
            warnings.append(
                "sharded_across_ranks payload does not allow scalability claim"
            )

    if semantics == "rank_local_replicated_kernel" and world_size > 1:
        warnings.append(
            "rank-local kernels may benchmark throughput but not single-workload capacity"
        )

    if semantics in INCOMPLETE_DISTRIBUTION_SEMANTICS and not payload.get(
        "scalability_blockers"
    ):
        warnings.append(
            "hybrid or incomplete distributed payload should report scalability_blockers"
        )

    if claim and evidence_type not in RELEASE_CLAIM_EVIDENCE_TYPES:
        warnings.append(
            "scalability_claim_allowed without release claim evidence is not release-grade"
        )

    release_evidence_errors: tuple[str, ...] = ()
    if claim and evidence_type in RELEASE_CLAIM_EVIDENCE_TYPES:
        release_evidence_errors = (
            _release_capacity_evidence_errors(payload)
            + _release_training_evidence_errors(payload)
            + _release_transport_evidence_errors(payload)
        )
        if release_evidence_errors:
            warnings.append(
                "release gate evidence is incomplete: "
                + "; ".join(dict.fromkeys(release_evidence_errors))
            )

    release_gate_allowed = bool(
        claim
        and not errors
        and not release_evidence_errors
        and semantics in SCALABLE_DISTRIBUTION_SEMANTICS
        and evidence_type in RELEASE_CLAIM_EVIDENCE_TYPES
    )

    return DistributedScalabilityAudit(
        valid=not errors,
        scalability_claim_allowed=claim and not errors,
        distribution_semantics=semantics,
        claim_evidence_type=evidence_type,
        release_gate_allowed=release_gate_allowed,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def attach_distributed_scalability_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a payload copy with a machine-readable scalability audit."""

    out = dict(payload)
    out["scalability_audit"] = audit_distributed_scalability(payload).summary()
    return out


def _release_capacity_evidence_errors(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("single_gpu_expected_oom") is not True:
        errors.append("release gate requires single_gpu_expected_oom=True")
    if not _is_nonempty(payload.get("capacity_baseline_device")):
        errors.append("release gate requires capacity_baseline_device")
    if not _is_nonempty(payload.get("capacity_failure_reason")):
        errors.append("release gate requires capacity_failure_reason")
    return tuple(errors)


def _release_mps_training_evidence_errors(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    if _backend_family(payload) != "mps":
        return ()

    errors: list[str] = []
    world_size = int(payload.get("world_size", 0) or 0)
    if (
        _semantics(payload, "mps_forward_distribution_semantics")
        != "sharded_across_ranks"
    ):
        errors.append("MPS release gate requires sharded forward evidence")
    if (
        _semantics(payload, "mps_backward_distribution_semantics")
        != "sharded_across_ranks"
    ):
        errors.append("MPS release gate requires sharded backward evidence")
    backward_execution = str(payload.get("backward_execution", "unknown")).lower()
    if not backward_execution or any(
        token in backward_execution
        for token in (
            "unknown",
            "pending",
            "planned",
            "not_executed",
            "local_simulated",
        )
    ):
        errors.append("MPS release gate requires executed production backward evidence")
    if not _mps_evidence_reported(payload.get("site_shard_ownership")):
        errors.append("MPS release gate requires site shard ownership")
    if not _mps_evidence_reported(payload.get("bond_shard_ownership")):
        errors.append("MPS release gate requires bond shard ownership")
    if not _mps_evidence_reported(payload.get("parameter_gradient_ownership")):
        errors.append("MPS release gate requires parameter-gradient ownership")
    if not _mps_evidence_reported(payload.get("boundary_gradient_ownership")):
        errors.append("MPS release gate requires boundary-gradient ownership")

    boundary_exchange = payload.get("boundary_adjoint_exchange")
    boundary_status = (
        str(boundary_exchange.get("status", "")).lower()
        if isinstance(boundary_exchange, Mapping)
        else ""
    )
    if boundary_status not in {
        "executed",
        "production_executed",
        "accelerator_executed",
    }:
        errors.append(
            "MPS release gate requires executed boundary-adjoint exchange evidence"
        )

    memory_plan = payload.get("mps_backward_memory_plan")
    memory_status = (
        str(memory_plan.get("status", "")).lower()
        if isinstance(memory_plan, Mapping)
        else ""
    )
    if not _mps_memory_plan_reported(
        memory_plan,
        world_size=world_size,
    ) or memory_status not in {"measured", "production_measured", "executed"}:
        errors.append(
            "MPS release gate requires production-measured per-rank "
            "backward memory evidence"
        )

    communication_plan = payload.get("mps_backward_communication_plan")
    communication_status = (
        str(communication_plan.get("status", "")).lower()
        if isinstance(communication_plan, Mapping)
        else ""
    )
    communication_executed = bool(
        _mps_communication_plan_reported(communication_plan)
        and communication_status
        in {
            "executed",
            "production_executed",
            "multi_node_production_transport",
        }
        and all(
            str(edge.get("execution_status", "")).lower() == "executed"
            for edge in communication_plan.get("boundary_edges", ())
            if isinstance(edge, Mapping)
        )
    )
    if not communication_executed:
        errors.append(
            "MPS release gate requires production-executed backward "
            "communication evidence"
        )

    fallback = str(payload.get("fallback_semantics", "unknown")).lower()
    if fallback not in {"none", "no_fallback", "not_used"}:
        errors.append(
            "MPS release gate rejects local replay, replicated autograd, "
            "and statevector fallback"
        )
    if _payload_blockers(payload):
        errors.append("MPS release gate requires blockers to be empty")
    return tuple(errors)


def _release_training_evidence_errors(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    optimizer_semantics = str(payload.get("optimizer_update_semantics", ""))
    if optimizer_semantics != "sharded_across_ranks":
        errors.append(
            "release gate requires optimizer_update_semantics='sharded_across_ranks'"
        )
    if _training_step_count(payload) <= 0:
        errors.append("release gate requires training_step_count > 0")
    gradient_semantics = str(
        payload.get(
            "gradient_distribution_semantics",
            payload.get("distribution_semantics", ""),
        )
    )
    if gradient_semantics != "sharded_across_ranks":
        errors.append(
            "release gate requires gradient_distribution_semantics='sharded_across_ranks'"
        )
    if not _ownership_sharded(
        payload,
        "parameter_ownership_semantics",
        "parameter_ownership",
    ):
        errors.append(
            "release gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership"
        )
    if not _ownership_sharded(
        payload,
        "gradient_ownership_semantics",
        "gradient_ownership",
    ):
        errors.append(
            "release gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership"
        )
    if not _ownership_sharded(
        payload,
        "optimizer_update_ownership_semantics",
        "optimizer_update_ownership",
    ):
        errors.append(
            "release gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership"
        )
    errors.extend(_release_mps_training_evidence_errors(payload))
    return tuple(errors)


def attach_sharded_optimizer_step_evidence(
    payload: Mapping[str, Any],
    *,
    training_step_count: int,
    parameter_ownership: Any,
    gradient_ownership: Any,
    optimizer_update_ownership: Any,
    optimizer_name: str = "explicit_sharded_optimizer_step",
) -> dict[str, Any]:
    """Attach explicit sharded optimizer-step ownership evidence to a payload.

    The helper does not turn preflight or runtime summaries into release
    evidence by itself. It only records that the caller can prove parameter,
    gradient, and optimizer-update ownership for executed training steps.
    """

    try:
        steps = int(training_step_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_step_count must be a positive integer") from exc
    if steps <= 0:
        raise ValueError("training_step_count must be a positive integer")
    if not _is_nonempty(parameter_ownership):
        raise ValueError("parameter_ownership must be nonempty")
    if not _is_nonempty(gradient_ownership):
        raise ValueError("gradient_ownership must be nonempty")
    if not _is_nonempty(optimizer_update_ownership):
        raise ValueError("optimizer_update_ownership must be nonempty")

    out = dict(payload)
    out.update(
        {
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "training_step_count": steps,
            "parameter_ownership": parameter_ownership,
            "gradient_ownership": gradient_ownership,
            "optimizer_update_ownership": optimizer_update_ownership,
            "optimizer_step_evidence": {
                "optimizer_name": str(optimizer_name),
                "training_step_count": steps,
                "parameter_ownership_semantics": "sharded_across_ranks",
                "gradient_ownership_semantics": "sharded_across_ranks",
                "optimizer_update_semantics": "sharded_across_ranks",
                "optimizer_update_ownership_semantics": "sharded_across_ranks",
                "parameter_ownership": parameter_ownership,
                "gradient_ownership": gradient_ownership,
                "optimizer_update_ownership": optimizer_update_ownership,
            },
        }
    )
    return out


def require_distributed_scalability(
    payload: Mapping[str, Any],
) -> DistributedScalabilityAudit:
    """Require that a payload is valid evidence for sharded scalability.

    This is the fail-closed guardrail for benchmark promotion and CI. It raises
    if the payload is replicated, incomplete, missing evidence, or otherwise not
    allowed to make a single-workload scalability claim.
    """

    audit = validate_distributed_claim_evidence(payload)
    if not audit.valid or not audit.release_gate_allowed:
        raise DistributedScalabilityError(audit)
    return audit


def validate_distributed_claim_evidence(
    payload: Mapping[str, Any],
) -> DistributedScalabilityAudit:
    """Validate release-grade evidence for a distributed scalability claim.

    Plan summaries, preflight checks, and development smoke tests may be honest
    sharding evidence, but they are not release-grade scalability evidence.
    This is the single release gate used by ``require_distributed_scalability``.
    """

    audit = audit_distributed_scalability(payload)
    errors = list(audit.errors)
    warnings = list(audit.warnings)
    if not audit.scalability_claim_allowed:
        errors.append("release gate requires scalability_claim_allowed=True")
    if audit.claim_evidence_type not in RELEASE_CLAIM_EVIDENCE_TYPES:
        errors.append(
            "release gate requires claim_evidence_type='production_training_benchmark' or 'release_payload'"
        )
    if audit.claim_evidence_type == "release_payload" and not (
        payload.get("release_payload") or payload.get("release_gate_payload")
    ):
        errors.append("release_payload evidence must be explicit")
    errors.extend(_release_capacity_evidence_errors(payload))
    errors.extend(_release_training_evidence_errors(payload))
    errors.extend(_release_transport_evidence_errors(payload))
    release_gate_allowed = bool(not errors and audit.release_gate_allowed)
    return DistributedScalabilityAudit(
        valid=not errors,
        scalability_claim_allowed=audit.scalability_claim_allowed and not errors,
        distribution_semantics=audit.distribution_semantics,
        claim_evidence_type=audit.claim_evidence_type,
        release_gate_allowed=release_gate_allowed,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
