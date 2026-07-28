"""Audit helpers for distributed scalability semantics.

These helpers enforce the product rule that multi-rank execution is only a
scalability result when one logical workload is actually sharded across ranks.
They are intentionally lightweight so runtime summaries, benchmark payloads,
and tests can share the same checks.
"""

# ruff: noqa: F401

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


def _release_policy_call(name: str, *args: Any, **kwargs: Any) -> Any:
    module = import_module("flagquantum.runtime.audit.release_policy")
    return getattr(module, name)(*args, **kwargs)


def _claim_evidence_type(payload: Mapping[str, Any]) -> str:
    return _release_policy_call("_claim_evidence_type", payload)


def _release_capacity_evidence_errors(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    return _release_policy_call("_release_capacity_evidence_errors", payload)


def _release_training_evidence_errors(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    return _release_policy_call("_release_training_evidence_errors", payload)


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
        CLAIMABILITY_STATUSES,
        EVIDENCE_BACKEND_FAMILIES,
        INCOMPLETE_DISTRIBUTION_SEMANTICS,
        MULTI_NODE_TRANSPORT_BACKENDS,
        RELEASE_CLAIM_EVIDENCE_TYPES,
        REPLICATED_DISTRIBUTION_SEMANTICS,
        SCALABLE_DISTRIBUTION_SEMANTICS,
        SINGLE_DEVICE_DISTRIBUTION_SEMANTICS,
        TRANSPORT_EVIDENCE_STATUSES,
    )


def _has_any(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in payload and payload[key] is not None for key in keys)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (str, bytes)):
        return bool(value)
    try:
        return len(value) > 0
    except TypeError:
        return True


def _has_nonempty(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in payload and _is_nonempty(payload[key]) for key in keys)


def _unique(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    try:
        iterable = tuple(values)
    except TypeError:
        iterable = (values,)
    return tuple(dict.fromkeys(str(item) for item in iterable if str(item)))


def _rank_memory_reported(payload: Mapping[str, Any]) -> bool:
    if _is_nonempty(payload.get("local_memory_bytes_by_rank")) or _is_nonempty(
        payload.get("rank_peak_memory_bytes")
    ):
        return True
    for key in ("rank_shards", "shards"):
        ranks = payload.get(key)
        if not _is_nonempty(ranks):
            continue
        for rank in ranks:
            if not isinstance(rank, Mapping):
                continue
            if any(
                memory_key in rank and rank[memory_key] is not None
                for memory_key in (
                    "local_state_bytes",
                    "local_memory_bytes",
                    "local_tensor_bytes",
                    "partial_bytes",
                )
            ):
                return True
    return False


def _communication_evidence_reported(payload: Mapping[str, Any]) -> bool:
    if _is_nonempty(payload.get("communication_tiers")):
        return True
    if _is_nonempty(payload.get("communication_plan")):
        return True
    return _has_any(
        payload,
        (
            "intra_node_communication_bytes",
            "inter_node_communication_bytes",
            "communication_bytes",
            "estimated_transfer_bytes",
            "forward_communication_bytes_per_rank_max",
            "backward_communication_bytes_per_rank_max",
        ),
    )


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _int_field(payload: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return int(default)


def _backend_family(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("state_mode", payload.get("mode", ""))).lower()
    if value in {"distributed_statevector", "jax_sharded_statevector", "statevector"}:
        return "statevector"
    if value in {"distributed_mps", "jax_sharded_mps", "mps"}:
        return "mps"
    if value in {
        "distributed_tensor_network",
        "jax_sharded_tensor_network",
        "tensor_network",
        "tn",
    }:
        return "tensor_network"
    return "unknown"


def _payload_blockers(payload: Mapping[str, Any]) -> tuple[str, ...]:
    blockers: list[str] = []
    for key in (
        "scalability_blockers",
        "blockers",
        "static_blockers",
        "device_blockers",
        "gradient_blockers",
        "production_blockers",
    ):
        blockers.extend(_unique(payload.get(key)))
    return tuple(dict.fromkeys(blockers))


def _rank_ownership_evidence(payload: Mapping[str, Any]) -> Any:
    for key in ("rank_ownership", "rank_shards", "rank_placement", "shards"):
        value = payload.get(key)
        if _is_nonempty(value):
            return value
    local_memory = payload.get("local_memory_bytes_by_rank")
    if _is_nonempty(local_memory):
        return tuple(
            {"rank": rank, "local_memory_bytes": int(memory)}
            for rank, memory in enumerate(local_memory)
        )
    return ()


def _normalized_memory_plan(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    memory_plan = dict(_nested_mapping(payload, "memory_plan"))
    if "local_memory_bytes_by_rank" not in memory_plan and _is_nonempty(
        payload.get("local_memory_bytes_by_rank")
    ):
        memory_plan["local_memory_bytes_by_rank"] = tuple(
            payload.get("local_memory_bytes_by_rank", ())
        )
    if "rank_memory_reported" not in memory_plan:
        memory_plan["rank_memory_reported"] = _rank_memory_reported(payload)
    for key in (
        "peak_intermediate_bytes",
        "communication_buffer_count",
        "peak_buffer_bytes",
    ):
        if key in payload and key not in memory_plan:
            memory_plan[key] = payload[key]
    return memory_plan


def _normalized_communication_plan(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    communication_plan = dict(_nested_mapping(payload, "communication_plan"))
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    if communication_tiers and "communication_tiers" not in communication_plan:
        communication_plan["communication_tiers"] = dict(communication_tiers)
    for key in (
        "communication_bytes",
        "estimated_transfer_bytes",
        "intra_node_communication_bytes",
        "inter_node_communication_bytes",
        "collective_counts",
        "p2p_counts",
    ):
        if key in payload and key not in communication_plan:
            communication_plan[key] = payload[key]
    if "communication_evidence_reported" not in communication_plan:
        communication_plan["communication_evidence_reported"] = (
            _communication_evidence_reported(payload)
        )
    return communication_plan


def _first_nonempty_value(*values: Any) -> Any:
    for value in values:
        if _is_nonempty(value):
            return value
    return None


def _transport_route(payload: Mapping[str, Any]) -> Any:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    return _first_nonempty_value(
        payload.get("executed_collective_route"),
        payload.get("collective_route"),
        payload.get("transport_route"),
        communication_plan.get("executed_collective_route"),
        communication_plan.get("collective_route"),
        communication_plan.get("transport_route"),
        communication_tiers.get("executed_collective_route"),
        communication_tiers.get("collective_route"),
        communication_tiers.get("transport_route"),
    )


def _transport_backend(payload: Mapping[str, Any]) -> str:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    backend = _first_nonempty_value(
        payload.get("network_backend"),
        payload.get("transport_backend"),
        payload.get("collective_backend"),
        communication_plan.get("network_backend"),
        communication_plan.get("transport_backend"),
        communication_plan.get("collective_backend"),
        communication_tiers.get("network_backend"),
        communication_tiers.get("transport_backend"),
        communication_tiers.get("collective_backend"),
    )
    return str(backend or "").lower()


def _topology_scope(payload: Mapping[str, Any]) -> str:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    scope = _first_nonempty_value(
        payload.get("topology_scope"),
        payload.get("transport_scope"),
        payload.get("collective_scope"),
        payload.get("transport_evidence_status"),
        communication_plan.get("topology_scope"),
        communication_plan.get("transport_scope"),
        communication_plan.get("collective_scope"),
        communication_tiers.get("topology_scope"),
        communication_tiers.get("transport_scope"),
        communication_tiers.get("collective_scope"),
    )
    return str(scope or "").lower()


def _route_scope(route: Any) -> str:
    if isinstance(route, Mapping):
        scope = _first_nonempty_value(
            route.get("topology_scope"), route.get("route_scope"), route.get("scope")
        )
        return str(scope or "").lower()
    return ""


def _route_backend(route: Any) -> str:
    if isinstance(route, Mapping):
        backend = _first_nonempty_value(
            route.get("network_backend"),
            route.get("transport_backend"),
            route.get("collective_backend"),
            route.get("backend"),
        )
        return str(backend or "").lower()
    return ""


def _communication_bytes_reported(payload: Mapping[str, Any]) -> bool:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    for source in (payload, communication_plan, communication_tiers):
        if _has_any(
            source,
            (
                "communication_bytes",
                "estimated_transfer_bytes",
                "intra_node_communication_bytes",
                "inter_node_communication_bytes",
            ),
        ):
            return True
    return False


def _rank_placement_reported(payload: Mapping[str, Any]) -> bool:
    return _is_nonempty(payload.get("rank_placement")) or _is_nonempty(
        payload.get("rank_ownership")
    )


def _topology_dependent_route(payload: Mapping[str, Any]) -> bool:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    text = " ".join(
        str(item)
        for item in (
            payload.get("topology_dependency", ""),
            payload.get("route_dependency", ""),
            communication_plan.get("topology_dependency", ""),
            communication_plan.get("route_dependency", ""),
            communication_tiers.get("topology_dependency", ""),
            communication_tiers.get("route_dependency", ""),
            communication_tiers.get("note", ""),
        )
    ).lower()
    return "topology" in text and (
        "depend" in text or "runtime" in text or "lowering" in text
    )


def evaluate_distributed_transport_evidence(
    payload: Mapping[str, Any],
) -> DistributedTransportEvidence:
    """Classify communication evidence without inferring hidden physical routes."""

    node_count = _int_field(payload, "node_count", 1)
    backend = _transport_backend(payload)
    topology_scope = _topology_scope(payload)
    route = _transport_route(payload)
    route_scope = _route_scope(route)
    route_backend = _route_backend(route)
    production_backend = (
        backend in MULTI_NODE_TRANSPORT_BACKENDS
        or route_backend in MULTI_NODE_TRANSPORT_BACKENDS
    )
    communication_bytes = _communication_bytes_reported(payload)
    rank_placement = _rank_placement_reported(payload)
    topology_dependent = _topology_dependent_route(payload)
    blockers: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    multi_node_route = bool(
        node_count > 1
        and communication_bytes
        and rank_placement
        and production_backend
        and not topology_dependent
        and (
            topology_scope == "multi_node_production_transport"
            or route_scope == "multi_node_production_transport"
            or route_scope == "multi_node"
            or _is_nonempty(route)
        )
    )
    single_node_collective = bool(
        node_count <= 1
        and _communication_evidence_reported(payload)
        and (
            topology_scope in {"single_node_executed_collective", "single_node"}
            or "single_node" in backend
            or _claim_evidence_type(payload) in RELEASE_CLAIM_EVIDENCE_TYPES
            or "executed" in str(payload.get("communication_execution", "")).lower()
        )
    )

    if multi_node_route:
        status = "multi_node_production_transport"
    elif single_node_collective:
        status = "single_node_executed_collective"
    elif node_count > 1 and _communication_evidence_reported(payload):
        status = "topology_dependent_planning"
        blockers.append("multi_node_production_transport_evidence_missing")
        if not rank_placement:
            blockers.append("multi_node_rank_placement_required_for_transport_evidence")
        if not communication_bytes:
            blockers.append(
                "multi_node_communication_bytes_required_for_transport_evidence"
            )
        if not production_backend:
            blockers.append("multi_node_production_transport_backend_required")
    elif _communication_evidence_reported(payload):
        status = "topology_dependent_planning"
        blockers.append("executed_collective_route_evidence_missing")
    else:
        status = "blocked"
        errors.append("distributed transport evidence requires communication evidence")

    return DistributedTransportEvidence(
        status=status,
        node_count=node_count,
        network_backend=backend or "unknown",
        topology_scope=topology_scope or "unknown",
        route=route or (),
        communication_bytes_reported=communication_bytes,
        rank_placement_reported=rank_placement,
        topology_dependent=topology_dependent
        or status == "topology_dependent_planning",
        blockers=tuple(dict.fromkeys(blockers)),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _release_transport_evidence_errors(payload: Mapping[str, Any]) -> tuple[str, ...]:
    transport = evaluate_distributed_transport_evidence(payload)
    if transport.node_count <= 1:
        return ()
    if transport.status == "multi_node_production_transport":
        return ()
    return (
        "release gate requires multi_node_production_transport evidence for node_count > 1",
        *transport.errors,
        *transport.blockers,
    )


def _semantics(payload: Mapping[str, Any], key: str, fallback: str = "unknown") -> str:
    value = payload.get(key, fallback)
    return str(value if value is not None else fallback)


def _has_topology_counts(payload: Mapping[str, Any]) -> bool:
    try:
        return (
            int(payload.get("world_size", 0) or 0) > 1
            and int(payload.get("local_world_size", 0) or 0) > 0
            and int(payload.get("node_count", 0) or 0) > 0
        )
    except (TypeError, ValueError):
        return False


def _memory_plan_has_shard_and_buffer(payload: Mapping[str, Any]) -> bool:
    memory_plan = _nested_mapping(payload, "memory_plan")
    rank_memory = (
        memory_plan.get("per_rank_shard_bytes")
        or memory_plan.get("local_memory_bytes_by_rank")
        or payload.get("local_memory_bytes_by_rank")
    )
    has_rank_shard = _is_nonempty(rank_memory) or _rank_memory_reported(payload)
    has_buffer = any(
        key in memory_plan and memory_plan[key] is not None
        for key in (
            "communication_buffer_bytes",
            "peak_communication_buffer_bytes",
            "communication_buffer_count",
            "peak_buffer_bytes",
        )
    ) or any(
        key in payload and payload[key] is not None
        for key in ("peak_buffer_bytes", "communication_buffer_count")
    )
    return bool(has_rank_shard and has_buffer)


def _communication_plan_has_statevector_route(payload: Mapping[str, Any]) -> bool:
    communication_plan = _nested_mapping(payload, "communication_plan")
    communication_tiers = _nested_mapping(payload, "communication_tiers")
    if not communication_plan and not communication_tiers:
        return False
    text_parts = [
        communication_plan.get("communication_execution", ""),
        communication_plan.get("transport", ""),
        communication_plan.get("collective", ""),
        communication_plan.get("model", ""),
        communication_tiers.get("communication_execution", ""),
        communication_tiers.get("collective", ""),
        communication_tiers.get("model", ""),
        payload.get("communication_execution", ""),
    ]
    patterns = (
        _unique(communication_plan.get("transport_patterns"))
        + _unique(communication_plan.get("communication_patterns"))
        + _unique(communication_plan.get("collective_blockers"))
        + _unique(communication_tiers.get("communications"))
    )
    text = " ".join(str(item) for item in (*text_parts, *patterns))
    tokens = (
        "pair_exchange",
        "all_to_all",
        "all-to-all",
        "collective",
        "pmap",
        "shard_map",
        "amplitude_exchange",
    )
    return any(token in text for token in tokens)


def _optimizer_step_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested_mapping(payload, "optimizer_step_evidence")


def _ownership_semantics(payload: Mapping[str, Any], key: str) -> str:
    evidence = _optimizer_step_evidence(payload)
    return str(payload.get(key, evidence.get(key, "unknown")))


def _ownership_nonempty(payload: Mapping[str, Any], key: str) -> bool:
    evidence = _optimizer_step_evidence(payload)
    return _is_nonempty(payload.get(key)) or _is_nonempty(evidence.get(key))


def _ownership_sharded(
    payload: Mapping[str, Any], semantics_key: str, ownership_key: str
) -> bool:
    return _ownership_semantics(
        payload, semantics_key
    ) == "sharded_across_ranks" and _ownership_nonempty(payload, ownership_key)


def _training_step_count(payload: Mapping[str, Any]) -> int:
    evidence = _optimizer_step_evidence(payload)
    try:
        return int(
            payload.get("training_step_count", evidence.get("training_step_count", 0))
            or 0
        )
    except (TypeError, ValueError):
        return 0


def _looks_like_preflight(payload: Mapping[str, Any]) -> bool:
    return bool(
        _claim_evidence_type(payload) == "plan_preflight"
        or payload.get("planner")
        or str(payload.get("status", "")) == "planned_not_executed"
    )


def _looks_like_local_simulation(payload: Mapping[str, Any]) -> bool:
    policy = (
        payload.get("distributed_backend_policy") or payload.get("backend_policy") or {}
    )
    profile = str(policy.get("profile", "")) if isinstance(policy, Mapping) else ""
    return bool(
        _claim_evidence_type(payload) == "development_smoke"
        or profile == "development"
        or "local_simulated" in str(payload.get("backward_execution", ""))
        or "development" in str(payload.get("executor", ""))
        or "single_process_development_simulator" in _payload_blockers(payload)
    )


def evaluate_distributed_evidence_contract(
    payload: Mapping[str, Any],
) -> DistributedEvidenceContract:
    """Normalize distributed claimability evidence across statevector, MPS, and TN.

    The contract is metadata-only and fail-closed.  It does not replace the
    release gate; instead it gives statevector training/5/6 summaries a shared vocabulary for
    evidence tier, claimability status, blockers, rank ownership, memory, and
    communication.
    """

    evidence_type = _claim_evidence_type(payload)
    backend_family = _backend_family(payload)
    semantics = _semantics(payload, "distribution_semantics")
    intended_semantics = _semantics(
        payload, "intended_distribution_semantics", semantics
    )
    blockers = _payload_blockers(payload)
    rank_ownership = _rank_ownership_evidence(payload)
    memory_plan = _normalized_memory_plan(payload)
    communication_plan = _normalized_communication_plan(payload)
    transport_evidence = evaluate_distributed_transport_evidence(payload).summary()
    release_errors = (
        _release_capacity_evidence_errors(payload)
        + _release_training_evidence_errors(payload)
        + _release_transport_evidence_errors(payload)
        if evidence_type in RELEASE_CLAIM_EVIDENCE_TYPES
        else ()
    )
    checks = {
        "known_backend_family": backend_family in EVIDENCE_BACKEND_FAMILIES
        and backend_family != "unknown",
        "known_claim_evidence_type": evidence_type in CLAIM_EVIDENCE_TYPES
        and evidence_type != "unknown",
        "sharding_semantics_available": (
            semantics == "sharded_across_ranks"
            or (
                evidence_type == "plan_preflight"
                and intended_semantics == "sharded_across_ranks"
                and bool(payload.get("sharding_plan_available", True))
            )
        ),
        "not_replicated_or_rank_local": semantics
        not in REPLICATED_DISTRIBUTION_SEMANTICS,
        "topology_reported": _has_topology_counts(payload),
        "rank_ownership_reported": _is_nonempty(rank_ownership),
        "memory_plan_reported": _rank_memory_reported(payload)
        or bool(memory_plan.get("rank_memory_reported")),
        "communication_plan_reported": _communication_evidence_reported(payload),
        "multi_node_transport_evidence_complete": (
            transport_evidence["node_count"] <= 1
            or transport_evidence["status"] == "multi_node_production_transport"
            or evidence_type not in RELEASE_CLAIM_EVIDENCE_TYPES
        ),
        "release_claim_allowed": (
            evidence_type not in RELEASE_CLAIM_EVIDENCE_TYPES
            or payload.get("scalability_claim_allowed") is True
        ),
        "release_training_evidence_complete": not release_errors,
        "blockers_empty_for_release": not blockers,
    }
    messages = {
        "known_backend_family": "distributed evidence contract requires statevector, mps, or tensor_network state mode",
        "known_claim_evidence_type": "distributed evidence contract requires known claim_evidence_type",
        "sharding_semantics_available": "distributed evidence contract requires sharded_across_ranks semantics or sharded preflight intent",
        "not_replicated_or_rank_local": "distributed evidence contract rejects replicated or rank-local semantics",
        "topology_reported": "distributed evidence contract requires world_size, local_world_size, and node_count",
        "rank_ownership_reported": "distributed evidence contract requires rank ownership",
        "memory_plan_reported": "distributed evidence contract requires per-rank memory evidence",
        "communication_plan_reported": "distributed evidence contract requires communication evidence",
        "multi_node_transport_evidence_complete": "distributed evidence contract requires multi-node production transport evidence for multi-node release claims",
    }
    errors = [message for key, message in messages.items() if not checks[key]]
    if evidence_type in RELEASE_CLAIM_EVIDENCE_TYPES:
        if not checks["release_claim_allowed"]:
            errors.append(
                "distributed evidence contract requires scalability_claim_allowed=True for release evidence"
            )
        errors.extend(release_errors)
        if blockers:
            errors.append(
                "distributed evidence contract requires blockers to be empty for release evidence"
            )
        if semantics != "sharded_across_ranks":
            errors.append(
                "distributed evidence contract requires release evidence to report distribution_semantics='sharded_across_ranks'"
            )

    if errors:
        status = "blocked"
    elif evidence_type == "plan_preflight":
        status = "preflight_only"
    elif evidence_type == "development_smoke":
        status = "local_simulation"
    elif evidence_type in RELEASE_CLAIM_EVIDENCE_TYPES:
        status = "claimable_production_training"
    else:
        status = "blocked"

    claimable = status == "claimable_production_training"
    warnings: list[str] = []
    if evidence_type == "production_runtime" and not claimable:
        warnings.append(
            "production_runtime evidence is not release-grade training evidence"
        )
    if blockers and evidence_type not in RELEASE_CLAIM_EVIDENCE_TYPES:
        warnings.append("non-release evidence carries blockers and remains fail-closed")

    return DistributedEvidenceContract(
        contract_version="distributed_evidence_contract_v1",
        backend_family=backend_family,
        status=status,
        claimable_production_training=claimable,
        claim_evidence_type=evidence_type,
        distribution_semantics=semantics,
        blockers=blockers,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        checks=checks,
        rank_ownership=rank_ownership,
        memory_plan=memory_plan,
        communication_plan=communication_plan,
        transport_evidence=transport_evidence,
        fail_closed=not claimable,
    )


def attach_distributed_evidence_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a payload copy with normalized distributed evidence metadata."""

    out = dict(payload)
    contract = evaluate_distributed_evidence_contract(out).summary()
    out["distributed_evidence_contract"] = contract
    out["claimability_status"] = contract["status"]
    out["claimable_production_training"] = contract["claimable_production_training"]
    out["distributed_evidence_contract_version"] = contract["contract_version"]
    out["transport_evidence"] = contract["transport_evidence"]
    out["transport_evidence_status"] = contract["transport_evidence"]["status"]
    return out


def evaluate_statevector_training_claimability(
    payload: Mapping[str, Any],
) -> StatevectorTrainingClaimabilityGate:
    """Evaluate the production statevector training claimability gate.

    This is the single machine-readable gate for the question: can this payload
    claim production distributed statevector training?  It is metadata-only and
    fail-closed; missing or unknown fields become errors.
    """

    contract = evaluate_distributed_evidence_contract(payload)
    blockers = contract.blockers
    distribution_semantics = _semantics(payload, "distribution_semantics")
    forward_semantics = _semantics(
        payload, "forward_distribution_semantics", distribution_semantics
    )
    backward_semantics = _semantics(
        payload,
        "backward_distribution_semantics",
        _semantics(payload, "gradient_distribution_semantics"),
    )
    parameter_gradient_ready = bool(
        payload.get("parameter_gradient_ready") is True
        or payload.get("production_value_gradient_ready") is True
        or payload.get("gradient_ready") is True
    )
    optimizer_semantics = _semantics(payload, "optimizer_update_semantics")
    full_state_replay = bool(
        payload.get("backward_uses_full_state_replay")
        or payload.get("full_state_replay_for_backward")
        or "full_state_replay" in str(payload.get("backward_strategy", ""))
    )
    checks = {
        "statevector_path": _statevector_mode_allowed(payload),
        "distribution_semantics_sharded": distribution_semantics
        == "sharded_across_ranks",
        "forward_semantics_sharded": forward_semantics == "sharded_across_ranks",
        "backward_semantics_sharded": backward_semantics == "sharded_across_ranks",
        "parameter_gradient_available": parameter_gradient_ready,
        "no_full_state_replay": not full_state_replay,
        "parameter_ownership_sharded": _ownership_sharded(
            payload,
            "parameter_ownership_semantics",
            "parameter_ownership",
        ),
        "gradient_ownership_sharded": _ownership_sharded(
            payload,
            "gradient_ownership_semantics",
            "gradient_ownership",
        ),
        "optimizer_update_ownership_sharded": _ownership_sharded(
            payload,
            "optimizer_update_ownership_semantics",
            "optimizer_update_ownership",
        ),
        "optimizer_step_preserves_sharded_ownership": (
            optimizer_semantics == "sharded_across_ranks"
            and _ownership_sharded(
                payload,
                "optimizer_update_ownership_semantics",
                "optimizer_update_ownership",
            )
        ),
        "training_steps_executed": _training_step_count(payload) > 0,
        "rank_ownership_explicit": _has_nonempty(
            payload,
            ("rank_shards", "rank_ownership", "rank_placement"),
        )
        or _is_nonempty(contract.rank_ownership),
        "memory_plan_has_per_rank_shard_and_comm_buffer": _memory_plan_has_shard_and_buffer(
            payload
        ),
        "communication_plan_has_statevector_route": _communication_plan_has_statevector_route(
            payload
        ),
        "world_topology_reported": _has_topology_counts(payload),
        "blockers_empty": not blockers,
    }
    messages = {
        "statevector_path": (
            "statevector training gate requires state_mode or mode to be one of "
            "distributed_statevector, jax_sharded_statevector, statevector"
        ),
        "distribution_semantics_sharded": "statevector training gate requires distribution_semantics='sharded_across_ranks'",
        "forward_semantics_sharded": "statevector training gate requires forward_distribution_semantics='sharded_across_ranks'",
        "backward_semantics_sharded": "statevector training gate requires backward_distribution_semantics='sharded_across_ranks'",
        "parameter_gradient_available": "statevector training gate requires parameter_gradient_ready=True",
        "no_full_state_replay": "statevector training gate rejects full-state replay for backward or parameter gradients",
        "parameter_ownership_sharded": "statevector training gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership",
        "gradient_ownership_sharded": "statevector training gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership",
        "optimizer_update_ownership_sharded": "statevector training gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership",
        "optimizer_step_preserves_sharded_ownership": "statevector training gate requires optimizer_update_semantics='sharded_across_ranks'",
        "training_steps_executed": "statevector training gate requires training_step_count > 0",
        "rank_ownership_explicit": "statevector training gate requires explicit rank ownership",
        "memory_plan_has_per_rank_shard_and_comm_buffer": "statevector training gate requires memory_plan with per-rank shard and communication buffer",
        "communication_plan_has_statevector_route": "statevector training gate requires communication_plan with pair-exchange, all-to-all, or collective route",
        "world_topology_reported": "statevector training gate requires world_size, local_world_size, and node_count",
        "blockers_empty": "statevector training gate requires blockers to be empty",
    }
    errors = tuple(message for key, message in messages.items() if not checks[key])
    if _looks_like_local_simulation(payload):
        status = "local_simulation"
    elif _looks_like_preflight(payload):
        status = "preflight_only"
    elif errors or blockers:
        status = "blocked"
    else:
        status = "claimable_production_training"
    return StatevectorTrainingClaimabilityGate(
        status=status,
        claimable_production_training=status == "claimable_production_training",
        errors=errors,
        blockers=blockers,
        checks=checks,
    )


__all__ = [  # noqa: F822
    "DistributedScalabilityAudit",
    "DistributedScalabilityError",
    "StatevectorTrainingClaimabilityGate",
    "MPSBackwardReadinessGate",
    "DistributedEvidenceContract",
    "DistributedTransportEvidence",
    "SCALABLE_DISTRIBUTION_SEMANTICS",
    "CLAIMABILITY_STATUSES",
    "SINGLE_DEVICE_DISTRIBUTION_SEMANTICS",
    "REPLICATED_DISTRIBUTION_SEMANTICS",
    "INCOMPLETE_DISTRIBUTION_SEMANTICS",
    "CLAIM_EVIDENCE_TYPES",
    "RELEASE_CLAIM_EVIDENCE_TYPES",
    "TRANSPORT_EVIDENCE_STATUSES",
    "audit_distributed_scalability",
    "attach_distributed_scalability_audit",
    "attach_distributed_evidence_contract",
    "attach_sharded_optimizer_step_evidence",
    "require_distributed_scalability",
    "validate_distributed_claim_evidence",
    "evaluate_statevector_training_claimability",
    "evaluate_mps_backward_readiness",
    "evaluate_distributed_evidence_contract",
    "evaluate_distributed_transport_evidence",
]


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    for module_name in (
        "flagquantum.runtime.audit.mps_readiness",
        "flagquantum.runtime.audit.release_policy",
    ):
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Canonical schema contracts; retained here as identity-compatible aliases.
try:
    from .errors import DistributedScalabilityError  # noqa: E402,F401
except ImportError:  # legacy benchmark file-loader path
    from flagquantum.runtime.audit.errors import (  # type: ignore
        DistributedScalabilityError,
    )

try:
    from .schema import (
        DistributedEvidenceContract,
        DistributedScalabilityAudit,
        DistributedTransportEvidence,
        MPSBackwardReadinessGate,
        StatevectorTrainingClaimabilityGate,
    )  # noqa: E402,F401
except ImportError:  # legacy benchmark file-loader path
    from flagquantum.runtime.audit.schema import (  # type: ignore
        DistributedEvidenceContract,
        DistributedScalabilityAudit,
        DistributedTransportEvidence,
        MPSBackwardReadinessGate,
        StatevectorTrainingClaimabilityGate,
    )

try:
    from .validation_helpers import (
        _backend_family,
        _communication_bytes_reported,
        _communication_evidence_reported,
        _communication_plan_has_statevector_route,
        _first_nonempty_value,
        _has_any,
        _has_nonempty,
        _has_topology_counts,
        _int_field,
        _is_nonempty,
        _looks_like_local_simulation,
        _looks_like_preflight,
        _memory_plan_has_shard_and_buffer,
        _nested_mapping,
        _normalized_communication_plan,
        _normalized_memory_plan,
        _optimizer_step_evidence,
        _ownership_nonempty,
        _ownership_semantics,
        _rank_memory_reported,
        _rank_ownership_evidence,
        _rank_placement_reported,
        _route_backend,
        _route_scope,
        _semantics,
        _statevector_mode_allowed,
        _topology_dependent_route,
        _topology_scope,
        _training_step_count,
        _transport_backend,
        _transport_route,
        _unique,
    )
except ImportError:  # legacy benchmark file-loader path
    from flagquantum.runtime.audit.validation_helpers import (  # type: ignore
        _backend_family,
        _communication_bytes_reported,
        _communication_evidence_reported,
        _communication_plan_has_statevector_route,
        _first_nonempty_value,
        _has_any,
        _has_nonempty,
        _has_topology_counts,
        _int_field,
        _is_nonempty,
        _looks_like_local_simulation,
        _looks_like_preflight,
        _memory_plan_has_shard_and_buffer,
        _nested_mapping,
        _normalized_communication_plan,
        _normalized_memory_plan,
        _optimizer_step_evidence,
        _ownership_nonempty,
        _ownership_semantics,
        _rank_memory_reported,
        _rank_ownership_evidence,
        _rank_placement_reported,
        _route_backend,
        _route_scope,
        _semantics,
        _statevector_mode_allowed,
        _topology_dependent_route,
        _topology_scope,
        _training_step_count,
        _transport_backend,
        _transport_route,
        _unique,
    )

try:
    from .statistics import (
        attach_distributed_evidence_contract,
        attach_distributed_scalability_audit,
    )
except ImportError:  # legacy benchmark file-loader path
    from flagquantum.runtime.audit.statistics import (  # type: ignore
        attach_distributed_evidence_contract,
        attach_distributed_scalability_audit,
    )
