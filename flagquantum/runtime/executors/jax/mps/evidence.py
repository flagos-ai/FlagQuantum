"""Accelerator-backed MPS backward evidence collection and summaries."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ..common import communication_tier as _communication_tier
from ..release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from ..runtime_environment import _require_jax


def _mps_accelerator_probe_classification() -> dict[str, Any]:
    """Return the fail-closed artifact vocabulary for the constrained probe."""

    return {
        "claim_evidence_type": "development_smoke",
        "artifact_classification": "measured_accelerator_probe",
        "observation_status": "measured_accelerator_probe_observed",
        "memory_status": "probe_array_footprint_estimate",
    }


def _collect_jax_mps_accelerator_backward_evidence(
    *,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Execute a single-node accelerator MPS backward evidence probe."""

    jax, jnp = _require_jax()
    accelerator_devices = tuple(
        device
        for device in jax.local_devices()
        if str(getattr(device, "platform", "unknown")).lower() != "cpu"
    )
    resolved_world_size = int(world_size or len(accelerator_devices))
    if resolved_world_size < 2 or len(accelerator_devices) < resolved_world_size:
        raise RuntimeError(
            "MPS accelerator backward evidence requires at least two local non-CPU JAX devices"
        )
    if int(jax.process_count()) != 1:
        raise RuntimeError(
            "MPS accelerator backward evidence currently supports single-node local JAX devices only"
        )
    devices = accelerator_devices[:resolved_world_size]
    parameters = jnp.arange(1, resolved_world_size + 1, dtype=jnp.float32)
    left_to_right = tuple((rank, rank + 1) for rank in range(resolved_world_size - 1))
    right_to_left = tuple((rank + 1, rank) for rank in range(resolved_world_size - 1))

    def exchange(local_value: Any) -> tuple[Any, Any]:
        return (
            jax.lax.ppermute(local_value, "mps_rank", left_to_right),
            jax.lax.ppermute(local_value, "mps_rank", right_to_left),
        )

    exchange_step = jax.pmap(exchange, axis_name="mps_rank", devices=devices)

    def backward_step(local_parameter: Any) -> tuple[Any, Any]:
        def loss(parameter: Any) -> Any:
            left_boundary, right_boundary = exchange(parameter)
            return 0.5 * parameter * parameter + 0.125 * parameter * (
                left_boundary + right_boundary
            )

        return jax.value_and_grad(loss)(local_parameter)

    mapped_backward = jax.pmap(
        backward_step,
        axis_name="mps_rank",
        devices=devices,
    )

    def block_ready(tree: Any) -> None:
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    block_ready(exchange_step(parameters))
    block_ready(mapped_backward(parameters))
    communication_start = time.perf_counter()
    boundary_values = exchange_step(parameters)
    block_ready(boundary_values)
    communication_seconds = max(
        time.perf_counter() - communication_start,
        1e-12,
    )
    backward_start = time.perf_counter()
    values, gradients = mapped_backward(parameters)
    block_ready((values, gradients))
    backward_seconds = max(time.perf_counter() - backward_start, 1e-12)

    item_bytes = int(parameters.dtype.itemsize)
    rank_memory_bytes = 5 * item_bytes
    device_placement = tuple(
        {
            "rank": rank,
            "node_id": 0,
            "local_rank": rank,
            "process_index": int(jax.process_index()),
            "device_id": str(device),
            "platform": str(getattr(device, "platform", "unknown")),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
        }
        for rank, device in enumerate(devices)
    )
    site_ownership = tuple(
        {
            "rank": rank,
            "wires": (rank,),
            "site_range": (rank,),
            "device_id": placement["device_id"],
        }
        for rank, placement in enumerate(device_placement)
    )
    bond_ownership = tuple(
        {
            "boundary_edge_id": f"edge_{rank}:{rank}-{rank + 1}",
            "left_rank": rank,
            "right_rank": rank + 1,
            "left_wire": rank,
            "right_wire": rank + 1,
        }
        for rank in range(resolved_world_size - 1)
    )
    communication_bytes_per_edge = 2 * item_bytes
    boundary_edges = tuple(
        {
            **bond,
            "communication_bytes": communication_bytes_per_edge,
            "communication_primitive": "jax_lax_ppermute",
            "communication_backend": "xla_pmap_collective_permute",
            "topology_tier": "intra_node",
            "topology_route": "single_node_xla_accelerator",
            "topology_dependent": False,
            "execution_status": "executed",
        }
        for bond in bond_ownership
    )
    communication_bytes = sum(
        int(edge["communication_bytes"]) for edge in boundary_edges
    )
    gradient_ownership = tuple(
        {
            "rank": rank,
            "owner_rank": rank,
            "parameter_indices": (rank,),
            "parameter_id": f"probe_parameter_{rank}",
            "gradient_value": float(gradients[rank]),
            "ownership_semantics": "rank_owned_gradient",
            "device_id": device_placement[rank]["device_id"],
        }
        for rank in range(resolved_world_size)
    )
    rank_memory = tuple(
        {
            "rank": rank,
            "site_range": (rank,),
            "forward_tensor_bytes": item_bytes,
            "backward_adjoint_bytes": item_bytes,
            "boundary_gradient_buffer_bytes": 2 * item_bytes,
            "parameter_gradient_bytes": item_bytes,
            "canonicalization_temporary_bytes": 0,
            "truncation_temporary_bytes": 0,
            "estimated_peak_backward_bytes": rank_memory_bytes,
        }
        for rank in range(resolved_world_size)
    )
    classification = _mps_accelerator_probe_classification()
    memory_plan = {
        "status": classification["memory_status"],
        "measurement_method": "known_live_probe_arrays_times_dtype_itemsize",
        "device_allocator_peak_measured": False,
        "measurement_scope": "accelerator_probe_array_footprint",
        "rank_memory": rank_memory,
        "forward_tensor_bytes_by_rank": tuple(item_bytes for _ in devices),
        "backward_adjoint_bytes_by_rank": tuple(item_bytes for _ in devices),
        "boundary_gradient_buffer_bytes_by_rank": tuple(
            2 * item_bytes for _ in devices
        ),
        "canonicalization_temporary_bytes_by_rank": tuple(0 for _ in devices),
        "truncation_temporary_bytes_by_rank": tuple(0 for _ in devices),
        "per_rank_peak_bytes": tuple(rank_memory_bytes for _ in devices),
        "local_memory_bytes_by_rank": tuple(rank_memory_bytes for _ in devices),
        "communication_buffer_bytes": 2 * item_bytes * resolved_world_size,
        "execution_scope": "single_node_measured_accelerator_probe",
        "blockers": (),
    }
    communication_plan = {
        "status": "accelerator_probe_executed",
        "boundary_edge_count": len(boundary_edges),
        "boundary_edges": boundary_edges,
        "communication_bytes": communication_bytes,
        "estimated_transfer_bytes": communication_bytes,
        "intra_node_communication_bytes": communication_bytes,
        "inter_node_communication_bytes": 0,
        "communication_backend": "xla_pmap_collective_permute",
        "communication_time_seconds": communication_seconds,
        "local_world_size": resolved_world_size,
        "node_count": 1,
        "execution_scope": "single_node_measured_accelerator_probe",
        "blockers": (),
    }
    blockers = (
        "mps_accelerator_probe_not_full_sharded_backward_executor",
        "mps_canonicalization_pullback_not_executed",
        "mps_accelerator_sharded_optimizer_update_pending",
        "mps_accelerator_capacity_evidence_pending",
        "mps_single_node_accelerator_evidence_not_multi_node_transport",
    )
    payload = {
        "executor": "jax_pmap_mps_accelerator_backward_probe",
        "claim_evidence_type": classification["claim_evidence_type"],
        "artifact_classification": classification["artifact_classification"],
        "state_mode": "jax_sharded_mps",
        "backend": "jax",
        "accelerator_backed": True,
        "accelerator_evidence_scope": "mps_boundary_backward_probe",
        "distribution_semantics": "sharded_across_ranks",
        "intended_distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": False,
        "world_size": resolved_world_size,
        "local_world_size": resolved_world_size,
        "node_count": 1,
        "device_placement": device_placement,
        "rank_to_device": device_placement,
        "rank_ownership": site_ownership,
        "rank_shards": site_ownership,
        "site_shard_ownership": site_ownership,
        "bond_shard_ownership": bond_ownership,
        "parameter_gradient_ownership": gradient_ownership,
        "boundary_gradient_ownership": tuple(
            {
                "left_rank": edge["left_rank"],
                "right_rank": edge["right_rank"],
                "ownership_semantics": "adjacent_rank_boundary_adjoint",
            }
            for edge in boundary_edges
        ),
        "boundary_adjoint_exchange": {
            "status": "accelerator_executed",
            "communication_backend": "xla_pmap_collective_permute",
            "records": boundary_edges,
        },
        "boundary_gradient_routes": boundary_edges,
        "canonicalization_backward_strategy": {
            "status": "planned",
            "strategy": "cross_shard_canonicalization_pullback",
        },
        "truncation_gradient_metadata": {"status": "not_required"},
        "mps_backward_memory_plan": memory_plan,
        "mps_backward_communication_plan": communication_plan,
        "accelerator_memory_evidence": memory_plan,
        "boundary_communication_backend": "xla_pmap_collective_permute",
        "backward_execution": ("jax_pmap_mps_accelerator_boundary_adjoint_probe"),
        "backward_execution_time_seconds": backward_seconds,
        "communication_time_seconds": communication_seconds,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_semantics": "not_measured",
        "optimizer_update_ownership_semantics": "not_measured",
        "training_step_count": 0,
        "parameter_ownership": gradient_ownership,
        "gradient_ownership": gradient_ownership,
        "optimizer_update_ownership": (),
        "fallback_semantics": "none",
        "backward_values": tuple(float(value) for value in values),
        "backward_gradients": tuple(float(value) for value in gradients),
        "local_memory_bytes_by_rank": memory_plan["local_memory_bytes_by_rank"],
        "communication_bytes": communication_bytes,
        "executed_collective_route": {
            "route_scope": "single_node",
            "network_backend": "xla",
            "communication_backend": "xla_pmap_collective_permute",
            "communication_bytes": communication_bytes,
            "rank_placement": device_placement,
        },
        "scalability_blockers": blockers,
        "gradient_blockers": blockers,
        "blockers": blockers,
    }
    payload["accelerator_backward_evidence"] = {
        "status": classification["observation_status"],
        "artifact_classification": classification["artifact_classification"],
        "accelerator_backed": True,
        "device_placement": device_placement,
        "world_size": resolved_world_size,
        "local_world_size": resolved_world_size,
        "node_count": 1,
        "rank_to_device": device_placement,
        "boundary_communication_backend": "xla_pmap_collective_permute",
        "accelerator_memory_evidence": memory_plan,
        "backward_execution_time_seconds": backward_seconds,
        "communication_time_seconds": communication_seconds,
        "gradient_ownership_semantics": "sharded_across_ranks",
        "parameter_gradient_ownership": gradient_ownership,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "blockers": blockers,
    }
    return _attach_mps_backward_readiness(payload)


def _summarize_minimal_mps_measured_runtime_evidence(
    rank_memory: Sequence[Mapping[str, Any]],
    communication_events: Sequence[Mapping[str, Any]],
    *,
    execution_scope: str,
) -> dict[str, Any]:
    """Validate measured resource records for the constrained MPS runtime."""

    blockers: list[str] = []
    ranks = tuple(dict(record) for record in rank_memory)
    events = tuple(dict(record) for record in communication_events)
    required_memory = (
        "forward_tensor_bytes",
        "backward_adjoint_bytes",
        "boundary_buffer_bytes",
        "optimizer_state_bytes",
        "measured_peak_live_tensor_bytes",
    )
    if {int(record.get("rank", -1)) for record in ranks} != {0, 1}:
        blockers.append("mps_measured_runtime_rank_coverage_incomplete")
    for record in ranks:
        rank = int(record.get("rank", -1))
        for field in required_memory:
            if field not in record or int(record.get(field, -1)) < 0:
                blockers.append(f"mps_measured_runtime_memory_missing:{rank}:{field}")
        components = (
            sum(int(record.get(field, 0)) for field in required_memory[:-1])
            + int(record.get("parameter_gradient_bytes", 0))
            + int(record.get("optimizer_update_buffer_bytes", 0))
        )
        if int(record.get("measured_peak_live_tensor_bytes", 0)) < components:
            blockers.append(f"mps_measured_runtime_peak_inconsistent:{rank}")
    if len(events) != 2:
        blockers.append("mps_measured_runtime_communication_event_count_incomplete")
    required_event = (
        "communication_bytes",
        "communication_primitive",
        "source_rank",
        "target_rank",
        "elapsed_time_seconds",
        "topology_tier",
    )
    for index, event in enumerate(events):
        for field in required_event:
            value = event.get(field)
            missing = value is None or value == ""
            if field == "communication_bytes":
                missing = missing or int(value or 0) <= 0
            if field == "elapsed_time_seconds":
                missing = missing or float(value or 0.0) <= 0.0
            if missing:
                blockers.append(
                    f"mps_measured_runtime_communication_missing:{index}:{field}"
                )
    blockers_out = tuple(dict.fromkeys(blockers))
    shared_collective_times = tuple(
        max(0.0, float(event.get("elapsed_time_seconds", 0.0)))
        for event in events
        if event.get("timing_scope") == "shared_collective_wall_clock"
    )
    independent_times = tuple(
        max(0.0, float(event.get("elapsed_time_seconds", 0.0)))
        for event in events
        if event.get("timing_scope") != "shared_collective_wall_clock"
    )
    return {
        "status": "measured" if not blockers_out else "blocked",
        "valid": not blockers_out,
        "execution_scope": str(execution_scope),
        "measurement_method": "live_tensor_nbytes_and_synchronized_wall_clock",
        "rank_memory": ranks,
        "communication_events": events,
        "total_communication_bytes": sum(
            max(0, int(event.get("communication_bytes", 0))) for event in events
        ),
        "total_communication_time_seconds": sum(independent_times)
        + (max(shared_collective_times) if shared_collective_times else 0.0),
        "allocator_peak_measured": False,
        "measurement_limitations": (
            "live_tensor_peak_is_not_device_allocator_peak",
            "single_node_route_only",
            *(
                ("accelerator_collective_wall_clock_includes_rank_step_compute",)
                if "accelerator" in str(execution_scope)
                else ()
            ),
        ),
        "blockers": blockers_out,
    }


def _build_mps_backward_resource_evidence(
    rank_summaries: Sequence[Mapping[str, Any]],
    boundary_protocols: Sequence[Mapping[str, Any]],
    boundary_exchange_records: Sequence[Mapping[str, Any]],
    parameter_gradient_bytes_by_rank: Sequence[int],
    truncation_records: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    local_world_size: int,
    node_count: int,
    execution_scope: str,
    communication_executed: bool,
) -> dict[str, Any]:
    """Account for rank-owned MPS backward memory and boundary traffic."""

    world_size = int(world_size)
    summaries = {int(item.get("rank", -1)): dict(item) for item in rank_summaries}
    memory_blockers = []
    communication_blockers = []
    if set(summaries) != set(range(world_size)):
        memory_blockers.append("mps_backward_resource_rank_coverage_incomplete")
    parameter_bytes = tuple(int(value) for value in parameter_gradient_bytes_by_rank)
    if len(parameter_bytes) != world_size:
        memory_blockers.append("mps_backward_parameter_gradient_memory_incomplete")
        parameter_bytes = tuple(0 for _ in range(world_size))
    boundary_buffers = [0 for _ in range(world_size)]
    records_by_edge = {}
    for value in boundary_exchange_records:
        record = dict(value)
        edge_id = str(record.get("boundary_edge_id", ""))
        records_by_edge.setdefault(edge_id, []).append(record)
        target = int(record.get("target_rank", -1))
        if 0 <= target < world_size:
            boundary_buffers[target] += int(record.get("local_memory_bytes", 0) or 0)

    edges = []
    intra_bytes = 0
    inter_bytes = 0
    for index, value in enumerate(boundary_protocols):
        protocol = dict(value)
        try:
            left_rank = int(protocol["left_rank"])
            right_rank = int(protocol["right_rank"])
            left_wire = int(protocol["left_wire"])
            right_wire = int(protocol["right_wire"])
        except (KeyError, TypeError, ValueError):
            communication_blockers.append(
                f"mps_backward_communication_route_missing:edge_{index}"
            )
            continue
        edge_id = f"edge_{index}:{left_wire}-{right_wire}"
        records = tuple(records_by_edge.get(edge_id, ()))
        tier = str(
            protocol.get(
                "tier",
                _communication_tier(
                    left_rank, right_rank, local_world_size=int(local_world_size)
                ),
            )
        )
        primitives = {
            str(item.get("communication_primitive", ""))
            for item in records
            if item.get("communication_primitive")
        }
        edge_bytes = (
            sum(int(item.get("communication_bytes", 0) or 0) for item in records)
            if communication_executed
            else int(protocol.get("estimated_transfer_bytes", 0) or 0)
        )
        primitive = (
            next(iter(primitives), "unknown")
            if communication_executed
            else "planned_adjacent_rank_point_to_point"
        )
        if communication_executed and len(records) != 2:
            communication_blockers.append(
                f"mps_backward_communication_direction_coverage_incomplete:{edge_id}"
            )
        if communication_executed and len(primitives) != 1:
            communication_blockers.append(
                f"mps_backward_communication_primitive_missing:{edge_id}"
            )
        if tier not in {"intra_node", "inter_node"}:
            communication_blockers.append(
                f"mps_backward_communication_topology_tier_missing:{edge_id}"
            )
        if edge_bytes <= 0:
            communication_blockers.append(
                f"mps_backward_communication_bytes_missing:{edge_id}"
            )
        route = (
            "topology_dependent"
            if tier == "inter_node"
            else (
                "single_node_local_cpu"
                if communication_executed
                else "planned_intra_node_route"
            )
        )
        if tier == "inter_node":
            inter_bytes += edge_bytes
        else:
            intra_bytes += edge_bytes
        edges.append(
            {
                "boundary_edge_id": edge_id,
                "left_rank": left_rank,
                "right_rank": right_rank,
                "left_wire": left_wire,
                "right_wire": right_wire,
                "communication_bytes": edge_bytes,
                "communication_primitive": primitive,
                "topology_tier": tier,
                "topology_route": route,
                "topology_dependent": route == "topology_dependent",
                "execution_status": (
                    "executed" if communication_executed else "planned_not_executed"
                ),
            }
        )
    if len(edges) != len(boundary_protocols):
        communication_blockers.append(
            "mps_backward_communication_edge_coverage_incomplete"
        )

    truncation_bonds = {int(item.get("bond", -1)) for item in truncation_records}
    rank_memory = []
    vectors = {
        name: []
        for name in ("forward", "backward", "canonicalization", "truncation", "peak")
    }
    for rank in range(world_size):
        summary = summaries.get(rank, {})
        wires = tuple(int(wire) for wire in summary.get("wires", ()))
        forward = int(summary.get("local_tensor_bytes", 0) or 0)
        tensor_bytes = tuple(
            int(value) for value in summary.get("tensor_bytes_by_wire", {}).values()
        )
        largest = max(tensor_bytes, default=forward)
        backward = forward
        canonicalization = 2 * largest
        truncation = 2 * largest if truncation_bonds.intersection(wires) else 0
        peak = (
            forward
            + backward
            + boundary_buffers[rank]
            + parameter_bytes[rank]
            + canonicalization
            + truncation
        )
        if forward <= 0:
            memory_blockers.append(f"mps_forward_tensor_memory_missing:rank_{rank}")
        vectors["forward"].append(forward)
        vectors["backward"].append(backward)
        vectors["canonicalization"].append(canonicalization)
        vectors["truncation"].append(truncation)
        vectors["peak"].append(peak)
        rank_memory.append(
            {
                "rank": rank,
                "site_range": wires,
                "forward_tensor_bytes": forward,
                "backward_adjoint_bytes": backward,
                "boundary_gradient_buffer_bytes": boundary_buffers[rank],
                "parameter_gradient_bytes": parameter_bytes[rank],
                "canonicalization_temporary_bytes": canonicalization,
                "truncation_temporary_bytes": truncation,
                "estimated_peak_backward_bytes": peak,
            }
        )

    memory_blockers = tuple(dict.fromkeys(memory_blockers))
    communication_blockers = tuple(dict.fromkeys(communication_blockers))
    memory_plan = {
        "status": "blocked" if memory_blockers else "complete_estimate",
        "rank_memory": tuple(rank_memory),
        "forward_tensor_bytes_by_rank": tuple(vectors["forward"]),
        "backward_adjoint_bytes_by_rank": tuple(vectors["backward"]),
        "boundary_gradient_buffer_bytes_by_rank": tuple(boundary_buffers),
        "canonicalization_temporary_bytes_by_rank": tuple(vectors["canonicalization"]),
        "truncation_temporary_bytes_by_rank": tuple(vectors["truncation"]),
        "per_rank_peak_bytes": tuple(vectors["peak"]),
        "local_memory_bytes_by_rank": tuple(vectors["peak"]),
        "communication_buffer_bytes": sum(boundary_buffers),
        "execution_scope": execution_scope,
        "blockers": memory_blockers,
    }
    communication_plan = {
        "status": (
            "blocked"
            if communication_blockers
            else (
                "not_required"
                if not boundary_protocols
                else (
                    "local_cpu_accounted"
                    if communication_executed
                    else "planned_not_executed"
                )
            )
        ),
        "boundary_edge_count": len(boundary_protocols),
        "boundary_edges": tuple(edges),
        "communication_bytes": intra_bytes + inter_bytes,
        "estimated_transfer_bytes": intra_bytes + inter_bytes,
        "intra_node_communication_bytes": intra_bytes,
        "inter_node_communication_bytes": inter_bytes,
        "local_world_size": int(local_world_size),
        "node_count": int(node_count),
        "execution_scope": execution_scope,
        "blockers": communication_blockers,
    }
    blockers = tuple(dict.fromkeys((*memory_blockers, *communication_blockers)))
    return {
        "status": "blocked" if blockers else "complete",
        "valid": not blockers,
        "claim_evidence_type": (
            "development_smoke" if communication_executed else "plan_preflight"
        ),
        "mps_backward_memory_plan": memory_plan,
        "mps_backward_communication_plan": communication_plan,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "blockers": blockers,
    }
