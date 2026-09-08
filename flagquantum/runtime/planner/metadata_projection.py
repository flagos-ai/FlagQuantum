"""Project training preflight summaries into Runtime metadata."""

from __future__ import annotations

from typing import Any, Mapping


def _jax_statevector_runtime_metadata_from_training_summary(
    summary: Mapping[str, Any] | None,
    *,
    fallback_memory_plan: Mapping[str, Any],
    fallback_communication_plan: Mapping[str, Any],
    fallback_gradient_plan: Mapping[str, Any],
    fallback_deployment_plan: Mapping[str, Any],
    require_deployment: bool,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Project the JAX training preflight into runtime-selection candidate metadata."""

    if not summary or summary.get("status") == "unavailable":
        return (
            dict(fallback_memory_plan),
            dict(fallback_communication_plan),
            dict(fallback_gradient_plan),
            dict(fallback_deployment_plan),
            {
                "execution_plan": {
                    "runtime_tier": "blocked",
                    "executor": "jax_sharded_statevector_training_plan",
                    "claim_evidence_type": "plan_preflight",
                    "state_mode": "jax_sharded_statevector",
                    "status": "training_preflight_unavailable",
                },
                "statevector_training_claimability_status": "blocked",
                "claimable_production_training": False,
                "runtime_readiness_blockers": (
                    "jax_sharded_statevector_training_plan_unavailable",
                ),
            },
        )
    memory_plan = dict(summary.get("memory_plan", {}) or fallback_memory_plan)
    communication_plan = dict(
        summary.get("communication_plan", {}) or fallback_communication_plan
    )
    statevector_plan = dict(summary.get("statevector_plan", {}) or {})
    statevector_communication_plan = dict(
        statevector_plan.get("communication_plan", {}) or {}
    )
    if not communication_plan.get("transport_patterns"):
        communication_plan["transport_patterns"] = tuple(
            statevector_communication_plan.get("transport_patterns", ()) or ()
        )
    if not communication_plan.get("collective_blockers"):
        communication_plan["collective_blockers"] = tuple(
            statevector_communication_plan.get("collective_blockers", ()) or ()
        )
    if communication_plan.get("transport_patterns") or communication_plan.get(
        "collective_blockers"
    ):
        communication_plan["statevector_route_source"] = "distributed_statevector_plan"
    gradient_plan = dict(fallback_gradient_plan)
    deployment_plan = dict(fallback_deployment_plan)
    blockers = tuple(str(item) for item in summary.get("blockers", ()) or ())
    claimability_status = str(
        summary.get("statevector_training_claimability_status", "blocked")
    )
    claimable = bool(summary.get("claimable_production_training", False))
    world_size = int(summary.get("world_size", 1) or 1)
    local_world_size = int(summary.get("local_world_size", world_size) or 1)
    node_count = int(summary.get("node_count", 1) or 1)
    backward_execution = str(summary.get("backward_execution", "unknown"))
    rank_shards = tuple(summary.get("rank_shards", ()) or ())
    rank_ownership = tuple(
        {
            "rank": (
                int(item.get("rank", index)) if isinstance(item, Mapping) else index
            ),
            "node_id": (
                int(item.get("node_id", index // max(1, local_world_size)))
                if isinstance(item, Mapping)
                else index // max(1, local_world_size)
            ),
            "local_rank": (
                int(item.get("local_rank", index % max(1, local_world_size)))
                if isinstance(item, Mapping)
                else index % max(1, local_world_size)
            ),
            "ownership": "amplitude_or_qubit_address_shard",
            "amplitude_start": (
                int(item.get("amplitude_start", item.get("start", 0)))
                if isinstance(item, Mapping)
                else 0
            ),
            "amplitude_end": (
                int(item.get("amplitude_end", item.get("end", 0)))
                if isinstance(item, Mapping)
                else 0
            ),
        }
        for index, item in enumerate(rank_shards)
    )
    if node_count > 1:
        communication_plan["inter_node_route"] = "topology_dependent"
        communication_plan["collective_route_resolution"] = "topology_dependent"
        communication_plan["multi_node_transport_assumption"] = (
            "topology_dependent_nccl_gloo_or_xla_collective_route"
        )
    else:
        communication_plan.setdefault("inter_node_route", "single_node")
    communication_plan.setdefault("rank_ownership", rank_ownership)
    communication_plan.setdefault("world_size", world_size)
    communication_plan.setdefault("local_world_size", local_world_size)
    communication_plan.setdefault("node_count", node_count)
    readiness_blockers = tuple(
        dict.fromkeys(
            blockers
            + tuple(str(item) for item in summary.get("device_blockers", ()) or ())
            + (() if claimable else ("statevector_training_not_claimable",))
            + (
                ()
                if summary.get("optimizer_update_semantics") == "sharded_across_ranks"
                else ("optimizer_update_semantics_not_measured",)
            )
        )
    )
    runtime_tier = (
        "claimable_production_execution"
        if claimable
        else (
            "local_development_simulator"
            if claimability_status == "local_simulation"
            else (
                "planned_shard_map_execution"
                if "shard_map" in backward_execution
                else (
                    "planned_pmap_execution"
                    if "pmap" in backward_execution
                    else "blocked"
                )
            )
        )
    )
    gradient_plan.update(
        {
            "parameter_gradient_ready": bool(
                summary.get("parameter_gradient_ready", False)
            ),
            "gradient_distribution_semantics": str(
                summary.get("backward_distribution_semantics", "incomplete")
            ),
            "forward_distribution_semantics": str(
                summary.get("forward_distribution_semantics", "unknown")
            ),
            "backward_distribution_semantics": str(
                summary.get("backward_distribution_semantics", "incomplete")
            ),
            "optimizer_update_semantics": str(
                summary.get("optimizer_update_semantics", "unknown")
            ),
            "backward_uses_full_state_replay": bool(
                summary.get("backward_uses_full_state_replay", True)
            ),
            "blockers": blockers,
            "fail_closed": bool(blockers or not claimable),
        }
    )
    deployment_plan["runtime_readiness"] = (
        "claimable_production_training" if claimable else claimability_status
    )
    deployment_plan["readiness_blockers"] = readiness_blockers
    deployment_plan["blockers"] = (
        readiness_blockers if require_deployment or not claimable else ()
    )
    metadata = {
        "execution_plan": {
            "runtime_tier": runtime_tier,
            "executor": "jax_sharded_statevector_training_plan",
            "claim_evidence_type": str(
                summary.get("claim_evidence_type", "plan_preflight")
            ),
            "state_mode": "jax_sharded_statevector",
            "backend": str(summary.get("backend", "jax")),
            "interface": str(summary.get("interface", "torch")),
            "planned_backward_execution": backward_execution,
            "forward_distribution_semantics": str(
                summary.get("forward_distribution_semantics", "unknown")
            ),
            "backward_distribution_semantics": str(
                summary.get("backward_distribution_semantics", "incomplete")
            ),
            "production_training_preflight_ready": bool(
                summary.get("production_training_preflight_ready", False)
            ),
            "claimable_production_training": claimable,
        },
        "rank_ownership": rank_ownership,
        "rank_shards": rank_shards,
        "statevector_training_claimability_gate": dict(
            summary.get("statevector_training_claimability_gate", {}) or {}
        ),
        "statevector_training_claimability_status": claimability_status,
        "claimable_production_training": claimable,
        "forward_distribution_semantics": str(
            summary.get("forward_distribution_semantics", "unknown")
        ),
        "backward_distribution_semantics": str(
            summary.get("backward_distribution_semantics", "incomplete")
        ),
        "parameter_gradient_ready": bool(
            summary.get("parameter_gradient_ready", False)
        ),
        "optimizer_update_semantics": str(
            summary.get("optimizer_update_semantics", "unknown")
        ),
        "backward_uses_full_state_replay": bool(
            summary.get("backward_uses_full_state_replay", True)
        ),
        "runtime_readiness_blockers": readiness_blockers,
    }
    return memory_plan, communication_plan, gradient_plan, deployment_plan, metadata


def _jax_mps_runtime_metadata_from_training_summary(
    summary: Mapping[str, Any] | None,
    *,
    fallback_memory_plan: Mapping[str, Any],
    fallback_communication_plan: Mapping[str, Any],
    fallback_gradient_plan: Mapping[str, Any],
    fallback_deployment_plan: Mapping[str, Any],
    require_deployment: bool,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Project MPS backward-readiness evidence into runtime-selection plans."""

    if not summary or summary.get("status") == "unavailable":
        blockers: tuple[str, ...] = ("jax_sharded_mps_training_plan_unavailable",)
        metadata = {
            "execution_plan": {
                "runtime_tier": "blocked",
                "executor": "jax_sharded_mps_training_plan",
                "state_mode": "jax_sharded_mps",
                "status": "training_preflight_unavailable",
            },
            "mps_backward_readiness_status": "blocked",
            "mps_runtime_blockers": blockers,
            "runtime_readiness_blockers": blockers,
            "claimable_production_training": False,
        }
        return (
            dict(fallback_memory_plan),
            dict(fallback_communication_plan),
            dict(fallback_gradient_plan),
            dict(fallback_deployment_plan),
            metadata,
        )
    mps_readiness = dict(summary.get("mps_runtime_summary", {}) or {})
    gate = dict(summary.get("mps_backward_readiness_gate", {}) or {})
    status = str(
        mps_readiness.get(
            "status",
            summary.get("mps_backward_readiness_status", "blocked"),
        )
    )
    evidence_status = dict(mps_readiness.get("evidence_status", {}) or {})
    blockers = tuple(
        dict.fromkeys(
            (
                *(str(item) for item in summary.get("blockers", ()) or ()),
                *(str(item) for item in summary.get("mps_runtime_blockers", ()) or ()),
            )
        )
    )
    claimable = bool(gate.get("production_training_claimable", False))
    topology = {
        "world_size": int(summary.get("world_size", 1) or 1),
        "local_world_size": int(summary.get("local_world_size", 1) or 1),
        "node_count": int(summary.get("node_count", 1) or 1),
        "rank_ownership": tuple(summary.get("site_shard_ownership", ()) or ()),
    }
    memory = dict(summary.get("mps_backward_memory_plan", {}) or fallback_memory_plan)
    memory.update(topology)
    memory["evidence_status"] = evidence_status.get("backward_memory", "pending")
    communication = dict(
        summary.get("mps_backward_communication_plan", {})
        or fallback_communication_plan
    )
    communication.update(topology)
    communication.update(
        {
            "communication_semantics": "mps_boundary_adjoint_and_gradient_exchange",
            "evidence_status": evidence_status.get("backward_communication", "pending"),
            "boundary_adjoint_exchange": summary.get(
                "boundary_adjoint_exchange", "pending"
            ),
            "boundary_gradient_routes": summary.get("boundary_gradient_routes", ()),
            "blockers": blockers,
        }
    )

    gradient = dict(fallback_gradient_plan)
    gradient.update(
        {
            "mps_backward_readiness_status": status,
            "parameter_gradient_ready": bool(
                summary.get("parameter_gradient_ready", False)
            ),
            "parameter_gradient_ownership": summary.get(
                "parameter_gradient_ownership", ()
            ),
            "parameter_gradient_ownership_evidence": summary.get(
                "parameter_gradient_ownership_evidence", {}
            ),
            "boundary_gradient_ownership": summary.get(
                "boundary_gradient_ownership", "unknown"
            ),
            "forward_distribution_semantics": str(
                summary.get(
                    "mps_forward_distribution_semantics",
                    summary.get("intended_distribution_semantics", "unknown"),
                )
            ),
            "backward_distribution_semantics": str(
                summary.get("mps_backward_distribution_semantics", "incomplete")
            ),
            "optimizer_update_semantics": str(
                summary.get("optimizer_update_semantics", "not_measured")
            ),
            "optimizer_update_ownership": summary.get("optimizer_update_ownership", ()),
            "evidence_status": evidence_status,
            "blockers": blockers,
            "fail_closed": not claimable,
        }
    )

    deployment = dict(fallback_deployment_plan)
    deployment["runtime_readiness"] = status
    deployment["readiness_blockers"] = blockers
    deployment["blockers"] = blockers if require_deployment or not claimable else ()

    runtime_tier = (
        status
        if status
        in {
            "control_plane_ready",
            "backward_preflight_ready",
            "production_backward_evidence",
        }
        else "claimable_production_execution" if claimable else "blocked"
    )
    metadata = {
        "execution_plan": {
            "runtime_tier": runtime_tier,
            "executor": "jax_sharded_mps_training_plan",
            "claim_evidence_type": str(
                summary.get("claim_evidence_type", "plan_preflight")
            ),
            "state_mode": "jax_sharded_mps",
            "planned_backward_execution": str(
                summary.get("backward_execution", "unknown")
            ),
            "status": status,
            "claimable_production_training": claimable,
        },
        "rank_ownership": topology["rank_ownership"],
        "bond_shard_ownership": tuple(summary.get("bond_shard_ownership", ()) or ()),
        "mps_backward_readiness_gate": gate,
        "mps_backward_readiness_status": status,
        "mps_backward_readiness_blockers": tuple(
            summary.get("mps_backward_readiness_blockers", ()) or ()
        ),
        "mps_runtime_summary": mps_readiness,
        "mps_runtime_blockers": tuple(summary.get("mps_runtime_blockers", ()) or ()),
        "runtime_readiness_blockers": blockers,
        "claimable_production_training": claimable,
    }
    return memory, communication, gradient, deployment, metadata
