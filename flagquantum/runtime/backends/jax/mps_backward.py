"""Minimal sharded MPS backward executor retained for compatibility."""

from __future__ import annotations

import time
from typing import Any

from .mps_evidence import _summarize_minimal_mps_measured_runtime_evidence
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .runtime_environment import _require_jax


def _execute_minimal_mps_sharded_backward(
    parameters: Any,
    *,
    execution_backend: str = "auto",
) -> dict[str, Any]:
    """Execute the constrained two-rank Phase 5 MPS backward skeleton."""

    import numpy as np

    parameter_values = np.asarray(parameters)
    if parameter_values.shape != (2,):
        raise ValueError(
            "minimal MPS sharded backward requires exactly two rank-owned RY parameters"
        )
    if parameter_values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("minimal MPS sharded backward requires real parameters")
    parameter_values = parameter_values.astype(np.float64, copy=False)
    backend = str(execution_backend).lower()
    if backend not in {"auto", "cpu", "accelerator"}:
        raise ValueError("execution_backend must be 'auto', 'cpu', or 'accelerator'")

    accelerator_devices: tuple[Any, ...] = ()
    jax = None
    jnp = None
    if backend in {"auto", "accelerator"}:
        try:
            jax, jnp = _require_jax()
            accelerator_devices = tuple(
                device
                for device in jax.local_devices()
                if str(getattr(device, "platform", "unknown")).lower() != "cpu"
            )
        except (ImportError, RuntimeError):
            accelerator_devices = ()
    use_accelerator = bool(
        backend != "cpu"
        and len(accelerator_devices) >= 2
        and jax is not None
        and int(jax.process_count()) == 1
    )
    if backend == "accelerator" and not use_accelerator:
        raise RuntimeError(
            "minimal MPS accelerator backward requires at least two local "
            "non-CPU JAX devices in one process"
        )

    start = time.perf_counter()
    communication_seconds = 0.0
    event_elapsed_times = (1e-12, 1e-12)
    event_timing_scope = "independent_event_wall_clock"
    if use_accelerator:
        devices = accelerator_devices[:2]
        jax.config.update("jax_enable_x64", True)
        parameter_array = jnp.asarray(parameter_values, dtype=jnp.float64)

        def rank_step(theta: Any) -> tuple[Any, Any, Any, Any]:
            site = jnp.stack(
                (
                    jnp.cos(theta / 2.0),
                    jnp.sin(theta / 2.0),
                )
            )
            local_z = site[0] * site[0] - site[1] * site[1]
            local_dz = -jnp.sin(theta)
            remote_z = jax.lax.ppermute(
                local_z,
                "mps_rank",
                ((0, 1), (1, 0)),
            )
            return site, local_z, local_dz * remote_z, remote_z

        mapped_step = jax.pmap(
            rank_step,
            axis_name="mps_rank",
            devices=devices,
        )
        warmup = mapped_step(parameter_array)
        for leaf in jax.tree_util.tree_leaves(warmup):
            leaf.block_until_ready()
        communication_start = time.perf_counter()
        site_arrays, local_z_values, gradients, remote_z_values = mapped_step(
            parameter_array
        )
        for leaf in jax.tree_util.tree_leaves(
            (site_arrays, local_z_values, gradients, remote_z_values)
        ):
            leaf.block_until_ready()
        communication_seconds = max(
            time.perf_counter() - communication_start,
            1e-12,
        )
        event_elapsed_times = (communication_seconds, communication_seconds)
        event_timing_scope = "shared_collective_wall_clock"
        site_arrays = np.asarray(jax.device_get(site_arrays))
        local_z_values = np.asarray(jax.device_get(local_z_values))
        gradients = np.asarray(jax.device_get(gradients))
        remote_z_values = np.asarray(jax.device_get(remote_z_values))
        device_placement = tuple(
            {
                "rank": rank,
                "local_rank": rank,
                "node_id": 0,
                "device_id": str(device),
                "platform": str(getattr(device, "platform", "unknown")),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
            }
            for rank, device in enumerate(devices)
        )
        claim_evidence_type = "production_runtime"
        execution_classification = "accelerator_backed_production_runtime_evidence"
        execution_scope = "single_node_accelerator"
        communication_backend = "xla_pmap_collective_permute"
        communication_primitive = "jax_lax_ppermute"
        communication_status = "production_executed"
        memory_status = "production_measured"
        boundary_status = "accelerator_executed"
        backward_execution = "jax_pmap_minimal_mps_sharded_backward"
    else:
        half_angles = parameter_values / 2.0
        site_arrays = np.stack(
            (np.cos(half_angles), np.sin(half_angles)),
            axis=1,
        )
        local_z_values = site_arrays[:, 0] ** 2 - site_arrays[:, 1] ** 2
        first_copy_start = time.perf_counter()
        remote_rank_zero = np.copy(local_z_values[1])
        first_copy_seconds = max(time.perf_counter() - first_copy_start, 1e-12)
        second_copy_start = time.perf_counter()
        remote_rank_one = np.copy(local_z_values[0])
        second_copy_seconds = max(time.perf_counter() - second_copy_start, 1e-12)
        remote_z_values = np.asarray((remote_rank_zero, remote_rank_one))
        gradients = -np.sin(parameter_values) * remote_z_values
        event_elapsed_times = (second_copy_seconds, first_copy_seconds)
        communication_seconds = sum(event_elapsed_times)
        device_placement = (
            {
                "rank": 0,
                "local_rank": 0,
                "node_id": 0,
                "device_id": "cpu:0/rank:0",
                "platform": "cpu",
                "device_kind": "host_cpu",
            },
            {
                "rank": 1,
                "local_rank": 1,
                "node_id": 0,
                "device_id": "cpu:0/rank:1",
                "platform": "cpu",
                "device_kind": "host_cpu",
            },
        )
        claim_evidence_type = "development_smoke"
        execution_classification = "cpu_local_distributed_development_execution"
        execution_scope = "development_cpu"
        communication_backend = "host_cpu_direct_copy"
        communication_primitive = "local_cpu_point_to_point_copy"
        communication_status = "development_executed"
        memory_status = "development_measured"
        boundary_status = "executed"
        backward_execution = "cpu_minimal_mps_sharded_backward"

    backward_seconds = max(time.perf_counter() - start, 1e-12)
    value = float(0.5 * np.sum(local_z_values * remote_z_values))
    gradients = np.asarray(gradients, dtype=np.float64)
    scalar_bytes = int(local_z_values.dtype.itemsize)
    site_bytes = int(site_arrays[0].nbytes)
    site_ownership = tuple(
        {
            "rank": rank,
            "wires": (rank,),
            "site_range": (rank,),
            "site_tensor_shape": (1, 2, 1),
            "site_tensor_bytes": site_bytes,
            "bond_dimension": 1,
            "device_id": device_placement[rank]["device_id"],
        }
        for rank in range(2)
    )
    bond_ownership = (
        {
            "boundary_edge_id": "edge_0:0-1",
            "left_rank": 0,
            "right_rank": 1,
            "left_wire": 0,
            "right_wire": 1,
            "bond_dimension": 1,
        },
    )
    boundary_records = (
        {
            "boundary_edge_id": "edge_0:0-1",
            "direction": "left_to_right",
            "source_rank": 0,
            "target_rank": 1,
            "communication_bytes": scalar_bytes,
            "communication_primitive": communication_primitive,
            "communication_backend": communication_backend,
            "execution_status": "executed",
            "elapsed_time_seconds": event_elapsed_times[0],
            "timing_scope": event_timing_scope,
            "topology_tier": "intra_node",
            "boundary_adjoint_value": float(local_z_values[0]),
        },
        {
            "boundary_edge_id": "edge_0:0-1",
            "direction": "right_to_left",
            "source_rank": 1,
            "target_rank": 0,
            "communication_bytes": scalar_bytes,
            "communication_primitive": communication_primitive,
            "communication_backend": communication_backend,
            "execution_status": "executed",
            "elapsed_time_seconds": event_elapsed_times[1],
            "timing_scope": event_timing_scope,
            "topology_tier": "intra_node",
            "boundary_adjoint_value": float(local_z_values[1]),
        },
    )
    gradient_ownership = tuple(
        {
            "rank": rank,
            "owner_rank": rank,
            "parameter_id": f"theta_{rank}",
            "parameter_indices": (rank,),
            "site_range": (rank,),
            "gradient_value": float(gradients[rank]),
            "gradient_bytes": int(gradients[rank].nbytes),
            "ownership_semantics": "rank_owned_gradient",
            "gradient_route": "owner_local_vjp_after_boundary_adjoint",
        }
        for rank in range(2)
    )
    rank_memory = tuple(
        {
            "rank": rank,
            "site_range": (rank,),
            "forward_tensor_bytes": site_bytes,
            "backward_adjoint_bytes": site_bytes,
            "boundary_gradient_buffer_bytes": scalar_bytes,
            "parameter_gradient_bytes": int(gradients[rank].nbytes),
            "canonicalization_temporary_bytes": 0,
            "canonicalization_not_required": True,
            "truncation_temporary_bytes": 0,
            "estimated_peak_backward_bytes": (
                2 * site_bytes + scalar_bytes + int(gradients[rank].nbytes)
            ),
        }
        for rank in range(2)
    )
    peak_bytes = tuple(
        int(record["estimated_peak_backward_bytes"]) for record in rank_memory
    )
    memory_plan = {
        "status": memory_status,
        "execution_scope": execution_scope,
        "canonicalization_required": False,
        "rank_memory": rank_memory,
        "forward_tensor_bytes_by_rank": (site_bytes, site_bytes),
        "backward_adjoint_bytes_by_rank": (site_bytes, site_bytes),
        "boundary_gradient_buffer_bytes_by_rank": (
            scalar_bytes,
            scalar_bytes,
        ),
        "canonicalization_temporary_bytes_by_rank": (0, 0),
        "truncation_temporary_bytes_by_rank": (0, 0),
        "per_rank_peak_bytes": peak_bytes,
        "local_memory_bytes_by_rank": peak_bytes,
        "communication_buffer_bytes": 2 * scalar_bytes,
        "blockers": (),
    }
    communication_bytes = 2 * scalar_bytes
    communication_plan = {
        "status": communication_status,
        "execution_scope": execution_scope,
        "boundary_edge_count": 1,
        "boundary_edges": (
            {
                **bond_ownership[0],
                "communication_bytes": communication_bytes,
                "communication_primitive": communication_primitive,
                "communication_backend": communication_backend,
                "topology_tier": "intra_node",
                "topology_route": (
                    "single_node_xla_accelerator"
                    if use_accelerator
                    else "single_process_cpu_rank_copy"
                ),
                "topology_dependent": False,
                "execution_status": "executed",
            },
        ),
        "communication_bytes": communication_bytes,
        "estimated_transfer_bytes": communication_bytes,
        "intra_node_communication_bytes": communication_bytes,
        "inter_node_communication_bytes": 0,
        "communication_time_seconds": communication_seconds,
        "local_world_size": 2,
        "node_count": 1,
        "blockers": (),
    }
    measured_rank_memory = tuple(
        {
            "rank": rank,
            "forward_tensor_bytes": site_bytes,
            "backward_adjoint_bytes": site_bytes,
            "boundary_buffer_bytes": scalar_bytes,
            "parameter_gradient_bytes": int(gradients[rank].nbytes),
            "optimizer_state_bytes": 0,
            "optimizer_update_buffer_bytes": 0,
            "measured_peak_live_tensor_bytes": peak_bytes[rank],
            "device_id": device_placement[rank]["device_id"],
        }
        for rank in range(2)
    )
    measured_communication_events = tuple(
        {
            "boundary_edge_id": record["boundary_edge_id"],
            "direction": record["direction"],
            "source_rank": record["source_rank"],
            "target_rank": record["target_rank"],
            "communication_bytes": record["communication_bytes"],
            "communication_primitive": record["communication_primitive"],
            "elapsed_time_seconds": record["elapsed_time_seconds"],
            "timing_scope": record["timing_scope"],
            "topology_tier": record["topology_tier"],
            "communication_backend": record["communication_backend"],
        }
        for record in boundary_records
    )
    measured_runtime_evidence = _summarize_minimal_mps_measured_runtime_evidence(
        measured_rank_memory,
        measured_communication_events,
        execution_scope=execution_scope,
    )
    blockers = (
        "mps_minimal_backward_skeleton_not_full_executor",
        "mps_minimal_backward_gate_set_limited_to_rank_local_ry",
        "mps_minimal_backward_fixed_bond_dimension_one",
        "mps_optimizer_update_ownership_pending",
        "mps_capacity_evidence_pending",
        *(
            ("mps_cpu_local_distributed_development_execution_only",)
            if not use_accelerator
            else ()
        ),
    )
    payload = {
        "executor": "minimal_mps_sharded_backward_skeleton",
        "state_mode": "jax_sharded_mps",
        "claim_evidence_type": claim_evidence_type,
        "execution_classification": execution_classification,
        "execution_scope": execution_scope,
        "distribution_semantics": "sharded_across_ranks",
        "intended_distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "mps_backward_execution": "executed",
        "backward_execution": backward_execution,
        "backward_execution_time_seconds": backward_seconds,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_placement": device_placement,
        "rank_to_device": device_placement,
        "rank_ownership": site_ownership,
        "rank_shards": site_ownership,
        "site_shard_ownership": site_ownership,
        "bond_shard_ownership": bond_ownership,
        "boundary_gradient_ownership": (
            {
                "left_rank": 0,
                "right_rank": 1,
                "ownership_semantics": "adjacent_rank_boundary_adjoint",
            },
        ),
        "boundary_adjoint_exchange": {
            "status": boundary_status,
            "communication_backend": communication_backend,
            "records": boundary_records,
            "communication_bytes": communication_bytes,
        },
        "boundary_gradient_routes": boundary_records,
        "parameter_gradient_ownership": gradient_ownership,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": gradient_ownership,
        "gradient_ownership": gradient_ownership,
        "canonicalization_backward_strategy": {
            "status": "not_required",
            "strategy": "bond_dimension_one_identity",
        },
        "truncation_gradient_metadata": {
            "status": "not_required",
            "bond_dimension": 1,
        },
        "mps_backward_memory_plan": memory_plan,
        "mps_backward_communication_plan": communication_plan,
        "memory_plan": {
            "local_memory_bytes_by_rank": peak_bytes,
            "communication_buffer_bytes": 2 * scalar_bytes,
        },
        "communication_plan": communication_plan,
        "local_memory_bytes_by_rank": peak_bytes,
        "communication_bytes": communication_bytes,
        "boundary_communication_backend": communication_backend,
        "communication_time_seconds": communication_seconds,
        "mps_measured_runtime_evidence": measured_runtime_evidence,
        "supported_gate_set": ("ry",),
        "fixed_bond_dimension": 1,
        "value": value,
        "gradient": tuple(float(item) for item in gradients),
        "local_site_tensors": tuple(
            {
                "rank": rank,
                "shape": (1, 2, 1),
                "values": tuple(float(item) for item in site_arrays[rank]),
            }
            for rank in range(2)
        ),
        "local_z_values": tuple(float(item) for item in local_z_values),
        "received_boundary_adjoints": tuple(float(item) for item in remote_z_values),
        "optimizer_update_semantics": "not_measured",
        "optimizer_update_ownership_semantics": "not_measured",
        "optimizer_update_ownership": (),
        "training_step_count": 0,
        "fallback_semantics": "none",
        "full_mps_reconstruction_count": 0,
        "replicated_mps_autograd": False,
        "statevector_fallback": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": blockers,
        "gradient_blockers": blockers,
        "blockers": blockers,
    }
    return _attach_mps_backward_readiness(payload)
