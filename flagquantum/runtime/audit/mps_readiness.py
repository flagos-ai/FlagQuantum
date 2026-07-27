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
        MPS_STATE_MODES,
        RELEASE_CLAIM_EVIDENCE_TYPES,
    )


from .engine import (
    _is_nonempty,
    _looks_like_local_simulation,
    _semantics,
    _training_step_count,
    _unique,
    attach_distributed_evidence_contract,
    evaluate_distributed_evidence_contract,
)
from .schema import (
    MPSBackwardReadinessGate,
)


def _mps_mode_allowed(payload: Mapping[str, Any]) -> bool:
    state_mode = payload.get("state_mode", payload.get("mode"))
    return state_mode is not None and str(state_mode) in MPS_STATE_MODES


def _mps_evidence_reported(value: Any) -> bool:
    if not _is_nonempty(value):
        return False
    missing_values = {
        "unknown",
        "missing",
        "pending",
        "not_measured",
        "incomplete",
        "not_available",
        "planned_not_executed",
    }
    if isinstance(value, Mapping):
        status = str(value.get("status", "")).lower()
        return status not in missing_values
    if isinstance(value, (str, bytes)):
        return str(value).lower() not in missing_values
    return True


def _mps_memory_plan_reported(value: Any, *, world_size: int) -> bool:
    if not isinstance(value, Mapping) or not value or world_size <= 0:
        return False
    if str(value.get("status", "")).lower() == "blocked":
        return False
    vectors = (
        "forward_tensor_bytes_by_rank",
        "backward_adjoint_bytes_by_rank",
        "boundary_gradient_buffer_bytes_by_rank",
        "canonicalization_temporary_bytes_by_rank",
        "truncation_temporary_bytes_by_rank",
        "per_rank_peak_bytes",
    )
    if any(
        not isinstance(value.get(field), (tuple, list))
        or len(value[field]) != world_size
        for field in vectors
    ):
        return False
    rank_memory = value.get("rank_memory")
    if not isinstance(rank_memory, (tuple, list)) or len(rank_memory) != world_size:
        return False
    required = {
        "rank",
        "forward_tensor_bytes",
        "backward_adjoint_bytes",
        "boundary_gradient_buffer_bytes",
        "canonicalization_temporary_bytes",
        "truncation_temporary_bytes",
        "estimated_peak_backward_bytes",
    }
    if {
        int(item.get("rank", -1)) for item in rank_memory if isinstance(item, Mapping)
    } != set(range(world_size)):
        return False
    return all(
        isinstance(item, Mapping)
        and required.issubset(item)
        and int(item.get("forward_tensor_bytes", 0)) > 0
        and int(item.get("backward_adjoint_bytes", 0)) > 0
        and (
            int(item.get("canonicalization_temporary_bytes", 0)) > 0
            or (
                value.get("canonicalization_required") is False
                and item.get("canonicalization_not_required") is True
            )
        )
        and int(item.get("estimated_peak_backward_bytes", 0)) > 0
        for item in rank_memory
    )


def _mps_communication_plan_reported(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    status = str(value.get("status", "")).lower()
    if status == "blocked":
        return False
    edge_count = int(value.get("boundary_edge_count", -1))
    edges = value.get("boundary_edges")
    if status == "not_required":
        return edge_count == 0 and int(value.get("communication_bytes", 0)) == 0
    if (
        edge_count <= 0
        or not isinstance(edges, (tuple, list))
        or len(edges) != edge_count
    ):
        return False
    required = {
        "boundary_edge_id",
        "left_rank",
        "right_rank",
        "communication_bytes",
        "communication_primitive",
        "topology_tier",
        "topology_route",
        "execution_status",
    }
    return all(
        isinstance(edge, Mapping)
        and required.issubset(edge)
        and int(edge.get("communication_bytes", 0)) > 0
        and str(edge.get("communication_primitive", "")) not in {"", "unknown"}
        and str(edge.get("topology_tier", "")) in {"intra_node", "inter_node"}
        and str(edge.get("topology_route", "")) not in {"", "unknown"}
        for edge in edges
    )


def _mps_optimizer_ownership_map(
    value: Any,
    *,
    role: str,
    world_size: int,
) -> dict[str, int] | None:
    if not isinstance(value, (tuple, list)) or not value or world_size <= 1:
        return None
    owner_field = {
        "parameter": "parameter_owner_rank",
        "gradient": "gradient_owner_rank",
        "update": "update_owner_rank",
    }[role]
    ownership: dict[str, int] = {}
    for record_index, record in enumerate(value):
        if not isinstance(record, Mapping):
            return None
        try:
            owner_rank = int(
                record.get(
                    owner_field, record.get("owner_rank", record.get("rank", -1))
                )
            )
        except (TypeError, ValueError):
            return None
        if owner_rank < 0 or owner_rank >= world_size:
            return None
        if role == "update":
            route = str(record.get("writeback_route", "")).lower()
            if (
                route in {"", "unknown", "not_measured", "replicated"}
                or "replicated" in route
                or "all_rank" in route
            ):
                return None

        identifiers: list[str] = []
        if "parameter_id" in record:
            identifiers.append(str(record["parameter_id"]))
        elif "parameter_flat_index" in record:
            identifiers.append(f"flat:{int(record['parameter_flat_index'])}")
        elif isinstance(record.get("parameter_indices"), (tuple, list)):
            identifiers.extend(
                f"flat:{int(index)}" for index in record["parameter_indices"]
            )
        elif "parameter_start" in record and "parameter_end" in record:
            try:
                start = int(record["parameter_start"])
                end = int(record["parameter_end"])
            except (TypeError, ValueError):
                return None
            if start < 0 or end <= start:
                return None
            identifiers.extend(f"flat:{index}" for index in range(start, end))
        else:
            return None

        for identifier in identifiers:
            if identifier in ownership:
                return None
            ownership[identifier] = owner_rank
    return ownership or None


def _mps_optimizer_ownership_checks(
    payload: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, bool]:
    parameter_ownership = _mps_optimizer_ownership_map(
        payload.get("parameter_ownership"),
        role="parameter",
        world_size=world_size,
    )
    gradient_ownership = _mps_optimizer_ownership_map(
        payload.get("gradient_ownership"),
        role="gradient",
        world_size=world_size,
    )
    update_ownership = _mps_optimizer_ownership_map(
        payload.get("optimizer_update_ownership"),
        role="update",
        world_size=world_size,
    )
    aligned = bool(
        parameter_ownership
        and gradient_ownership
        and update_ownership
        and parameter_ownership == gradient_ownership == update_ownership
    )
    return {
        "optimizer_parameter_ownership_reported": parameter_ownership is not None,
        "optimizer_gradient_ownership_reported": gradient_ownership is not None,
        "optimizer_update_ownership_reported": update_ownership is not None,
        "optimizer_ownership_aligned": aligned,
    }


def evaluate_mps_backward_readiness(
    payload: Mapping[str, Any],
) -> MPSBackwardReadinessGate:
    """Evaluate MPS backward readiness without bypassing release audit."""

    contract = evaluate_distributed_evidence_contract(payload)
    input_blockers = contract.blockers
    distribution_semantics = _semantics(payload, "distribution_semantics")
    intended_semantics = _semantics(
        payload,
        "intended_distribution_semantics",
        distribution_semantics,
    )
    forward_semantics = _semantics(
        payload,
        "mps_forward_distribution_semantics",
        _semantics(payload, "forward_distribution_semantics", intended_semantics),
    )
    backward_semantics = _semantics(
        payload,
        "mps_backward_distribution_semantics",
        _semantics(
            payload,
            "backward_distribution_semantics",
            _semantics(payload, "gradient_distribution_semantics"),
        ),
    )
    site_ownership = payload.get("site_shard_ownership")
    bond_ownership = payload.get("bond_shard_ownership")
    parameter_gradient_ownership = payload.get("parameter_gradient_ownership")
    boundary_gradient_ownership = payload.get("boundary_gradient_ownership")
    boundary_adjoint_exchange = payload.get("boundary_adjoint_exchange")
    boundary_gradient_routes = payload.get("boundary_gradient_routes")
    canonicalization_strategy = payload.get("canonicalization_backward_strategy")
    truncation_metadata = payload.get("truncation_gradient_metadata")
    backward_memory_plan = payload.get("mps_backward_memory_plan")
    backward_communication_plan = payload.get("mps_backward_communication_plan")
    parameter_ownership_semantics = _semantics(payload, "parameter_ownership_semantics")
    gradient_ownership_semantics = _semantics(payload, "gradient_ownership_semantics")
    optimizer_semantics = _semantics(payload, "optimizer_update_semantics")
    optimizer_ownership_semantics = _semantics(
        payload, "optimizer_update_ownership_semantics"
    )
    parameter_ownership = payload.get("parameter_ownership")
    gradient_ownership = payload.get("gradient_ownership")
    optimizer_ownership = payload.get("optimizer_update_ownership")
    optimizer_checks = _mps_optimizer_ownership_checks(
        payload,
        world_size=int(payload.get("world_size", 0) or 0),
    )
    fallback_semantics = payload.get("fallback_semantics", "unknown")
    fallback_values = (
        tuple(str(key) for key in fallback_semantics)
        if isinstance(fallback_semantics, Mapping)
        else _unique(fallback_semantics)
    )
    fallback_text = " ".join(fallback_values).lower()
    forbidden_fallbacks = {
        "replicated_mps_autograd": "mps_replicated_autograd_not_claimable",
        "full_local_mps_state_view": "mps_full_local_state_view_not_claimable",
        "full_local_mps_replay": "mps_full_local_mps_replay_not_claimable",
        "statevector_fallback": "mps_statevector_fallback_not_claimable",
    }
    active_fallback_blockers = tuple(
        blocker
        for token, blocker in forbidden_fallbacks.items()
        if token in fallback_text
    )
    backward_execution = str(payload.get("backward_execution", "unknown")).lower()
    production_backward_execution = bool(
        contract.claim_evidence_type
        in {"production_runtime", *RELEASE_CLAIM_EVIDENCE_TYPES}
        and backward_execution
        and not any(
            token in backward_execution
            for token in (
                "unknown",
                "pending",
                "local_simulated",
                "planned",
                "not_executed",
            )
        )
    )
    memory_status = (
        str(backward_memory_plan.get("status", "")).lower()
        if isinstance(backward_memory_plan, Mapping)
        else ""
    )
    communication_status = (
        str(backward_communication_plan.get("status", "")).lower()
        if isinstance(backward_communication_plan, Mapping)
        else ""
    )
    production_memory_evidence = memory_status in {
        "measured",
        "production_measured",
        "executed",
    }
    production_communication_evidence = bool(
        communication_status
        in {
            "executed",
            "production_executed",
            "multi_node_production_transport",
        }
        and isinstance(backward_communication_plan, Mapping)
        and all(
            str(edge.get("execution_status", "")).lower() == "executed"
            for edge in backward_communication_plan.get("boundary_edges", ())
            if isinstance(edge, Mapping)
        )
    )

    checks = {
        "shared_evidence_contract_v1": contract.contract_version
        == "distributed_evidence_contract_v1",
        "mps_path": _mps_mode_allowed(payload),
        "forward_distribution_sharded": forward_semantics == "sharded_across_ranks",
        "backward_distribution_sharded": backward_semantics == "sharded_across_ranks",
        "site_shard_ownership_reported": _mps_evidence_reported(site_ownership),
        "bond_shard_ownership_reported": _mps_evidence_reported(bond_ownership),
        "parameter_gradient_ownership_reported": _mps_evidence_reported(
            parameter_gradient_ownership
        ),
        "boundary_gradient_ownership_reported": _mps_evidence_reported(
            boundary_gradient_ownership
        ),
        "boundary_adjoint_exchange_reported": _mps_evidence_reported(
            boundary_adjoint_exchange
        ),
        "boundary_gradient_routes_reported": _mps_evidence_reported(
            boundary_gradient_routes
        ),
        "canonicalization_backward_strategy_reported": _mps_evidence_reported(
            canonicalization_strategy
        ),
        "truncation_gradient_metadata_reported": _mps_evidence_reported(
            truncation_metadata
        ),
        "backward_memory_plan_reported": _mps_memory_plan_reported(
            backward_memory_plan,
            world_size=int(payload.get("world_size", 0) or 0),
        ),
        "backward_communication_plan_reported": _mps_communication_plan_reported(
            backward_communication_plan
        ),
        "optimizer_parameter_ownership_sharded": (
            parameter_ownership_semantics == "sharded_across_ranks"
        ),
        "optimizer_gradient_ownership_sharded": (
            gradient_ownership_semantics == "sharded_across_ranks"
        ),
        "optimizer_update_sharded": optimizer_semantics == "sharded_across_ranks",
        "optimizer_update_ownership_sharded": (
            optimizer_ownership_semantics == "sharded_across_ranks"
        ),
        **optimizer_checks,
        "optimizer_training_steps_executed": _training_step_count(payload) > 0,
        "fallback_semantics_reported": _mps_evidence_reported(fallback_semantics),
        "no_forbidden_fallback": not active_fallback_blockers,
        "not_local_simulation": not (
            _looks_like_local_simulation(payload) or "local_simulation" in fallback_text
        ),
        "production_backward_execution_reported": production_backward_execution,
        "production_memory_evidence_reported": production_memory_evidence,
        "production_communication_evidence_reported": production_communication_evidence,
        "input_blockers_empty": not input_blockers,
        "shared_release_contract_claimable": contract.claimable_production_training,
    }
    blocker_codes = {
        "shared_evidence_contract_v1": "mps_shared_evidence_contract_v1_required",
        "mps_path": "phase5_mps_state_mode_required",
        "forward_distribution_sharded": "mps_forward_distribution_not_sharded",
        "backward_distribution_sharded": "mps_backward_distribution_not_sharded",
        "site_shard_ownership_reported": "mps_backward_site_shard_ownership_incomplete",
        "bond_shard_ownership_reported": "mps_backward_bond_shard_ownership_incomplete",
        "parameter_gradient_ownership_reported": "mps_parameter_gradient_ownership_incomplete",
        "boundary_gradient_ownership_reported": "mps_boundary_gradient_ownership_incomplete",
        "boundary_adjoint_exchange_reported": "mps_boundary_gradient_exchange_pending",
        "boundary_gradient_routes_reported": "mps_boundary_gradient_routes_incomplete",
        "canonicalization_backward_strategy_reported": "mps_canonicalization_backward_strategy_pending",
        "truncation_gradient_metadata_reported": "mps_truncation_gradient_metadata_incomplete",
        "backward_memory_plan_reported": "mps_backward_memory_plan_incomplete",
        "backward_communication_plan_reported": "mps_backward_communication_plan_incomplete",
        "optimizer_parameter_ownership_sharded": "mps_parameter_ownership_semantics_not_measured",
        "optimizer_gradient_ownership_sharded": "mps_gradient_ownership_semantics_not_measured",
        "optimizer_update_sharded": "mps_optimizer_update_semantics_not_measured",
        "optimizer_update_ownership_sharded": "mps_optimizer_update_ownership_semantics_not_measured",
        "optimizer_parameter_ownership_reported": "mps_optimizer_parameter_ownership_incomplete",
        "optimizer_gradient_ownership_reported": "mps_optimizer_gradient_ownership_incomplete",
        "optimizer_update_ownership_reported": "mps_optimizer_update_ownership_incomplete",
        "optimizer_ownership_aligned": "mps_optimizer_ownership_not_aligned",
        "optimizer_training_steps_executed": "mps_optimizer_training_step_not_executed",
        "fallback_semantics_reported": "mps_fallback_semantics_unknown",
        "not_local_simulation": "mps_local_simulation_not_claimable",
        "production_backward_execution_reported": "mps_production_backward_execution_not_measured",
        "production_memory_evidence_reported": "mps_production_backward_memory_not_measured",
        "production_communication_evidence_reported": "mps_production_backward_communication_not_executed",
    }
    generated_blockers = tuple(
        code for check, code in blocker_codes.items() if not checks[check]
    )
    blockers = tuple(
        dict.fromkeys((*input_blockers, *active_fallback_blockers, *generated_blockers))
    )
    messages = {
        "shared_evidence_contract_v1": "MPS backward readiness gate requires distributed_evidence_contract_v1",
        "mps_path": "MPS backward readiness gate requires state_mode or mode to identify an MPS path",
        "forward_distribution_sharded": "MPS backward readiness gate requires sharded forward distribution semantics",
        "backward_distribution_sharded": "MPS backward readiness gate requires mps_backward_distribution_semantics='sharded_across_ranks'",
        "site_shard_ownership_reported": "MPS backward readiness gate requires site shard ownership",
        "bond_shard_ownership_reported": "MPS backward readiness gate requires bond shard ownership",
        "parameter_gradient_ownership_reported": "MPS backward readiness gate requires parameter-gradient ownership",
        "boundary_gradient_ownership_reported": "MPS backward readiness gate requires boundary-gradient ownership",
        "boundary_adjoint_exchange_reported": "MPS backward readiness gate requires a boundary-adjoint exchange plan",
        "boundary_gradient_routes_reported": "MPS backward readiness gate requires boundary-gradient routes",
        "canonicalization_backward_strategy_reported": "MPS backward readiness gate requires a canonicalization backward strategy",
        "truncation_gradient_metadata_reported": "MPS backward readiness gate requires truncation-gradient metadata",
        "backward_memory_plan_reported": "MPS backward readiness gate requires per-rank backward memory evidence",
        "backward_communication_plan_reported": "MPS backward readiness gate requires a backward communication plan",
        "optimizer_parameter_ownership_sharded": "MPS backward readiness gate requires sharded parameter ownership semantics",
        "optimizer_gradient_ownership_sharded": "MPS backward readiness gate requires sharded gradient ownership semantics",
        "optimizer_update_sharded": "MPS backward readiness gate requires sharded optimizer-update semantics",
        "optimizer_update_ownership_sharded": "MPS backward readiness gate requires sharded optimizer-update ownership semantics",
        "optimizer_parameter_ownership_reported": "MPS backward readiness gate requires optimizer parameter ownership",
        "optimizer_gradient_ownership_reported": "MPS backward readiness gate requires optimizer gradient ownership",
        "optimizer_update_ownership_reported": "MPS backward readiness gate requires optimizer-update ownership and writeback routes",
        "optimizer_ownership_aligned": "MPS backward readiness gate requires parameter, gradient, and update owners to align",
        "optimizer_training_steps_executed": "MPS backward readiness gate requires training_step_count > 0",
        "fallback_semantics_reported": "MPS backward readiness gate requires explicit fallback semantics",
        "no_forbidden_fallback": "MPS backward readiness gate rejects replicated autograd, full local replay, and statevector fallback",
        "not_local_simulation": "MPS backward readiness gate rejects local simulation as production evidence",
        "production_backward_execution_reported": "MPS backward readiness gate requires executed production backward evidence",
        "input_blockers_empty": "MPS backward readiness gate requires blockers to be empty for production claims",
    }
    errors = tuple(message for key, message in messages.items() if not checks[key])
    control_plane_ready = all(
        checks[key]
        for key in (
            "shared_evidence_contract_v1",
            "mps_path",
            "forward_distribution_sharded",
            "site_shard_ownership_reported",
            "bond_shard_ownership_reported",
            "fallback_semantics_reported",
            "no_forbidden_fallback",
            "not_local_simulation",
        )
    )
    backward_preflight_ready = control_plane_ready and all(
        checks[key]
        for key in (
            "backward_distribution_sharded",
            "parameter_gradient_ownership_reported",
            "boundary_gradient_ownership_reported",
            "boundary_adjoint_exchange_reported",
            "boundary_gradient_routes_reported",
            "canonicalization_backward_strategy_reported",
            "truncation_gradient_metadata_reported",
            "backward_memory_plan_reported",
            "backward_communication_plan_reported",
        )
    )
    production_training_claimable = bool(
        backward_preflight_ready
        and checks["production_backward_execution_reported"]
        and checks["production_memory_evidence_reported"]
        and checks["production_communication_evidence_reported"]
        and checks["optimizer_parameter_ownership_sharded"]
        and checks["optimizer_gradient_ownership_sharded"]
        and checks["optimizer_update_sharded"]
        and checks["optimizer_update_ownership_sharded"]
        and checks["optimizer_parameter_ownership_reported"]
        and checks["optimizer_gradient_ownership_reported"]
        and checks["optimizer_update_ownership_reported"]
        and checks["optimizer_ownership_aligned"]
        and checks["optimizer_training_steps_executed"]
        and checks["input_blockers_empty"]
        and checks["shared_release_contract_claimable"]
    )
    if production_training_claimable:
        status = "production_training_claimable"
    elif input_blockers or not control_plane_ready:
        status = "blocked"
    elif (
        backward_preflight_ready
        and checks["production_backward_execution_reported"]
        and checks["production_memory_evidence_reported"]
        and checks["production_communication_evidence_reported"]
    ):
        status = "production_backward_evidence"
    elif backward_preflight_ready:
        status = "backward_preflight_ready"
    else:
        status = "control_plane_ready"

    return MPSBackwardReadinessGate(
        status=status,
        production_training_claimable=production_training_claimable,
        fail_closed=not production_training_claimable,
        contract_version=contract.contract_version,
        claim_evidence_type=contract.claim_evidence_type,
        errors=errors,
        blockers=blockers,
        checks=checks,
        evidence={
            "mps_forward_distribution_semantics": forward_semantics,
            "mps_backward_distribution_semantics": backward_semantics,
            "site_shard_ownership": site_ownership,
            "bond_shard_ownership": bond_ownership,
            "parameter_gradient_ownership": parameter_gradient_ownership,
            "boundary_gradient_ownership": boundary_gradient_ownership,
            "boundary_adjoint_exchange": boundary_adjoint_exchange,
            "boundary_gradient_routes": boundary_gradient_routes,
            "canonicalization_backward_strategy": canonicalization_strategy,
            "truncation_gradient_metadata": truncation_metadata,
            "mps_backward_memory_plan": backward_memory_plan,
            "mps_backward_communication_plan": backward_communication_plan,
            "parameter_ownership_semantics": parameter_ownership_semantics,
            "gradient_ownership_semantics": gradient_ownership_semantics,
            "optimizer_update_semantics": optimizer_semantics,
            "optimizer_update_ownership_semantics": optimizer_ownership_semantics,
            "training_step_count": _training_step_count(payload),
            "parameter_ownership": parameter_ownership,
            "gradient_ownership": gradient_ownership,
            "optimizer_update_ownership": optimizer_ownership,
            "optimizer_step_evidence": payload.get("optimizer_step_evidence"),
            "fallback_semantics": fallback_semantics,
            "backward_execution": backward_execution,
            "mps_backward_memory_evidence_status": memory_status,
            "mps_backward_communication_evidence_status": communication_status,
            "shared_distributed_evidence_contract_status": contract.status,
        },
    )


def _mps_evidence_stage(value: Any, *, reported: bool) -> str:
    if not reported:
        return "pending"
    if isinstance(value, Mapping):
        status = str(value.get("status", "")).lower()
        if "plan" in status or "not_executed" in status or "estimate" in status:
            return "planned"
        if "executed" in status or status in {"measured", "production_measured"}:
            return "executed"
        if status:
            return status
    return "available"


def _attach_mps_runtime_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach an MPS runtime-readiness projection derived from the gate."""

    out = attach_distributed_evidence_contract(payload)
    gate = evaluate_mps_backward_readiness(out).summary()
    checks = dict(gate.get("checks", {}))
    fallback = out.get("fallback_semantics", "unknown")
    fallback_text = (
        " ".join(str(key) for key in fallback).lower()
        if isinstance(fallback, (tuple, list, set))
        else (
            " ".join(str(key) for key in fallback).lower()
            if isinstance(fallback, Mapping)
            else str(fallback).lower()
        )
    )

    precise_blockers: list[str] = []
    if not checks.get("boundary_adjoint_exchange_reported", False):
        precise_blockers.append("mps_boundary_adjoint_exchange_pending")
    if not checks.get("parameter_gradient_ownership_reported", False):
        precise_blockers.append("mps_parameter_gradient_ownership_pending")
    if not checks.get("backward_memory_plan_reported", False):
        precise_blockers.append("mps_backward_memory_evidence_pending")
    if not checks.get("backward_communication_plan_reported", False):
        precise_blockers.append("mps_backward_communication_evidence_pending")
    optimizer_checks = (
        "optimizer_parameter_ownership_sharded",
        "optimizer_gradient_ownership_sharded",
        "optimizer_update_sharded",
        "optimizer_update_ownership_sharded",
        "optimizer_parameter_ownership_reported",
        "optimizer_gradient_ownership_reported",
        "optimizer_update_ownership_reported",
        "optimizer_ownership_aligned",
        "optimizer_training_steps_executed",
    )
    if not all(checks.get(key, False) for key in optimizer_checks):
        precise_blockers.append("mps_optimizer_update_ownership_pending")
    if "full_local_mps" in fallback_text or "full_mps" in fallback_text:
        precise_blockers.append("mps_full_local_replay_not_claimable")
    if "replicated_mps_autograd" in fallback_text:
        precise_blockers.append("mps_replicated_autograd_not_claimable")

    superseded_gate_blockers = {
        "mps_boundary_gradient_exchange_pending",
        "mps_parameter_gradient_ownership_incomplete",
        "mps_backward_memory_plan_incomplete",
        "mps_backward_communication_plan_incomplete",
        "mps_parameter_ownership_semantics_not_measured",
        "mps_gradient_ownership_semantics_not_measured",
        "mps_optimizer_update_semantics_not_measured",
        "mps_optimizer_update_ownership_semantics_not_measured",
        "mps_optimizer_parameter_ownership_incomplete",
        "mps_optimizer_gradient_ownership_incomplete",
        "mps_optimizer_update_ownership_incomplete",
        "mps_optimizer_ownership_not_aligned",
        "mps_optimizer_training_step_not_executed",
        "mps_full_local_state_view_not_claimable",
        "mps_full_local_mps_replay_not_claimable",
        "mps_replicated_autograd_not_claimable",
    }
    runtime_blockers = tuple(
        dict.fromkeys(
            (
                *precise_blockers,
                *(
                    str(item)
                    for item in gate.get("blockers", ())
                    if str(item) not in superseded_gate_blockers
                ),
            )
        )
    )

    blocker_aliases = {
        "mps_production_boundary_adjoint_exchange_pending": "mps_boundary_adjoint_exchange_pending",
        "mps_boundary_gradient_exchange_pending": "mps_boundary_adjoint_exchange_pending",
        "mps_parameter_gradient_runtime_pending": "mps_parameter_gradient_ownership_pending",
        "mps_parameter_gradient_ownership_incomplete": "mps_parameter_gradient_ownership_pending",
        "mps_backward_memory_plan_incomplete": "mps_backward_memory_evidence_pending",
        "mps_backward_communication_plan_incomplete": "mps_backward_communication_evidence_pending",
        "mps_optimizer_update_semantics_not_measured": "mps_optimizer_update_ownership_pending",
        "mps_optimizer_update_ownership_incomplete": "mps_optimizer_update_ownership_pending",
        "full_local_mps_state_view": "mps_full_local_replay_not_claimable",
        "full_mps_sync_fallback": "mps_full_local_replay_not_claimable",
        "replicated_mps_autograd": "mps_replicated_autograd_not_claimable",
    }
    existing_blockers = tuple(
        dict.fromkeys(
            blocker_aliases.get(str(item), str(item))
            for item in (
                out.get("blockers", ()) or out.get("scalability_blockers", ()) or ()
            )
        )
    )
    is_parameter_flow = out.get("planner") == "jax_sharded_mps_parameter_flow"
    integrated_blockers = (
        existing_blockers
        if is_parameter_flow
        else tuple(dict.fromkeys((*existing_blockers, *runtime_blockers)))
    )
    if "blockers" in out or out.get("planner"):
        out["blockers"] = integrated_blockers
    out["scalability_blockers"] = integrated_blockers

    out = attach_distributed_evidence_contract(out)
    gate = evaluate_mps_backward_readiness(out).summary()
    checks = dict(gate.get("checks", {}))
    evidence_status = {
        "boundary_adjoint_exchange": _mps_evidence_stage(
            out.get("boundary_adjoint_exchange"),
            reported=bool(checks.get("boundary_adjoint_exchange_reported", False)),
        ),
        "parameter_gradient_ownership": _mps_evidence_stage(
            out.get("parameter_gradient_ownership_evidence")
            or out.get("parameter_gradient_ownership"),
            reported=bool(checks.get("parameter_gradient_ownership_reported", False)),
        ),
        "backward_memory": _mps_evidence_stage(
            out.get("mps_backward_memory_plan"),
            reported=bool(checks.get("backward_memory_plan_reported", False)),
        ),
        "backward_communication": _mps_evidence_stage(
            out.get("mps_backward_communication_plan"),
            reported=bool(checks.get("backward_communication_plan_reported", False)),
        ),
        "optimizer_update_ownership": (
            "executed"
            if checks.get("optimizer_ownership_aligned", False)
            and checks.get("optimizer_training_steps_executed", False)
            else "pending"
        ),
    }
    if (
        gate.get("claim_evidence_type") == "plan_preflight"
        and evidence_status["parameter_gradient_ownership"] == "available"
    ):
        evidence_status["parameter_gradient_ownership"] = "planned"
    runtime_summary = {
        "status": gate["status"],
        "claim_evidence_type": gate["claim_evidence_type"],
        "production_training_claimable": gate["production_training_claimable"],
        "fail_closed": gate["fail_closed"],
        "evidence_status": evidence_status,
        "runtime_blockers": runtime_blockers,
        "next_engineering_blockers": runtime_blockers,
        "gate_checks": checks,
    }
    out["mps_backward_readiness_gate"] = gate
    out["mps_backward_readiness_status"] = gate["status"]
    out["mps_backward_readiness_blockers"] = gate["blockers"]
    out["mps_runtime_summary"] = runtime_summary
    out["mps_runtime_blockers"] = runtime_blockers
    # Compatibility keys for historical benchmark and release payloads.
    out["phase5_mps_backward_readiness_gate"] = gate
    out["phase5_mps_backward_readiness_status"] = gate["status"]
    out["phase5_mps_backward_readiness_blockers"] = gate["blockers"]
    out["phase5_mps_runtime_summary"] = runtime_summary
    out["phase5_mps_runtime_blockers"] = runtime_blockers
    out["claimable_production_training"] = gate["production_training_claimable"]
    return out
