"""Dependency-free normalization helpers for audit payload validation."""

from typing import Any, Mapping


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


def _rank_memory_reported(payload: Mapping[str, Any]) -> bool:
    if _is_nonempty(payload.get("local_memory_bytes_by_rank")):
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
        ),
    )


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
    return any(
        token in text
        for token in (
            "pair_exchange",
            "all_to_all",
            "all-to-all",
            "collective",
            "pmap",
            "shard_map",
            "amplitude_exchange",
        )
    )


def _optimizer_step_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested_mapping(payload, "optimizer_step_evidence")


def _ownership_semantics(payload: Mapping[str, Any], key: str) -> str:
    evidence = _optimizer_step_evidence(payload)
    return str(payload.get(key, evidence.get(key, "unknown")))


def _ownership_nonempty(payload: Mapping[str, Any], key: str) -> bool:
    evidence = _optimizer_step_evidence(payload)
    return _is_nonempty(payload.get(key)) or _is_nonempty(evidence.get(key))


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
    blockers = tuple(
        blocker
        for key in (
            "scalability_blockers",
            "blockers",
            "static_blockers",
            "device_blockers",
            "gradient_blockers",
            "production_blockers",
        )
        for blocker in _unique(payload.get(key))
    )
    return bool(
        _claim_evidence_type(payload) == "development_smoke"
        or profile == "development"
        or "local_simulated" in str(payload.get("backward_execution", ""))
        or "development" in str(payload.get("executor", ""))
        or "single_process_development_simulator" in blockers
    )


def _claim_evidence_type(payload: Mapping[str, Any]) -> str:
    return _semantics(payload, "claim_evidence_type")


def _statevector_mode_allowed(payload: Mapping[str, Any]) -> bool:
    state_mode = payload.get("state_mode")
    if state_mode is not None:
        return str(state_mode) in {
            "distributed_statevector",
            "jax_sharded_statevector",
            "statevector",
        }
    mode = payload.get("mode")
    return mode is not None and str(mode) in {
        "distributed_statevector",
        "jax_sharded_statevector",
        "statevector",
    }


def _mps_mode_allowed(payload: Mapping[str, Any]) -> bool:
    return _semantics(payload, "state_mode") in {
        "distributed_mps",
        "jax_sharded_mps",
        "mps",
    }


def _mps_evidence_reported(value: Any) -> bool:
    return _is_nonempty(value) and isinstance(value, Mapping)


def _mps_memory_plan_reported(value: Any, *, world_size: int) -> bool:
    return (
        isinstance(value, Mapping)
        and world_size > 0
        and _is_nonempty(value.get("shard"))
    )


def _mps_communication_plan_reported(value: Any) -> bool:
    return isinstance(value, Mapping) and _is_nonempty(value.get("route"))


def _mps_optimizer_ownership_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested_mapping(payload, "optimizer_ownership")


def _mps_optimizer_ownership_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    ownership = _mps_optimizer_ownership_map(payload)
    return {
        "ownership_reported": bool(ownership),
        "parameter_owner_reported": _is_nonempty(ownership.get("parameter_owner")),
        "gradient_owner_reported": _is_nonempty(ownership.get("gradient_owner")),
    }


def _mps_evidence_stage(value: Any, *, reported: bool) -> str:
    if not reported:
        return "missing"
    if isinstance(value, Mapping):
        return str(value.get("stage", "reported"))
    return "reported"


__all__ = [
    name for name in globals() if name.startswith("_") and not name.startswith("__")
]
