"""Typed plan assembly for runtime candidates."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class CircuitAnalysisView(Protocol):
    @property
    def n_wires(self) -> int: ...

    @property
    def depth(self) -> int: ...

    @property
    def n_instructions(self) -> int: ...

    @property
    def two_qubit_gates(self) -> int: ...

    @property
    def multi_qubit_gates(self) -> int: ...

    @property
    def has_noise(self) -> bool: ...


def memory_plan(
    *,
    mode: str,
    state: str,
    memory: int,
    single_device_bytes: int,
    memory_limit_bytes: int | None,
    world_size: int,
    distribution_semantics: str,
    peak_intermediate_bytes: int | None,
) -> dict[str, Any]:
    memory = int(memory)
    limit = None if memory_limit_bytes is None else int(memory_limit_bytes)
    return {
        "mode": mode,
        "state_mode": state,
        "distribution_semantics": distribution_semantics,
        "estimated_state_bytes": memory,
        "per_rank_state_bytes": memory,
        "single_device_reference_bytes": int(single_device_bytes),
        "peak_intermediate_bytes": int(peak_intermediate_bytes or memory),
        "memory_limit_bytes": limit,
        "fits_memory_limit": True if limit is None else memory <= limit,
        "single_device_expected_oom": (
            False if limit is None else int(single_device_bytes) > limit
        ),
        "estimate_source": "static_flagquantum_ir_planner",
    }


def communication_plan(
    *,
    mode: str,
    state: str,
    analysis: CircuitAnalysisView,
    world_size: int,
    local_world_size: int,
    node_count: int,
    distribution_semantics: str,
    memory: int,
    rank_ownership: tuple[dict[str, Any], ...],
    complex_bytes: int,
) -> dict[str, Any]:
    if distribution_semantics == "single_device_fast_path":
        return {
            "communication_semantics": "none",
            "world_size": 1,
            "local_world_size": 1,
            "node_count": 1,
            "estimated_communication_bytes": 0,
            "collective_counts": {},
            "p2p_counts": {},
            "rank_ownership": (),
            "topology_dependency": "none",
        }
    if distribution_semantics == "rank_local_replicated_kernel":
        return {
            "communication_semantics": "rank_local_replicated_kernel",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "estimated_communication_bytes": 0,
            "collective_counts": {},
            "p2p_counts": {},
            "rank_ownership": rank_ownership,
            "topology_dependency": "none_for_quantum_state",
        }
    if state == "statevector" and distribution_semantics == "sharded_across_ranks":
        distributed_gate_count = max(
            0, int(analysis.two_qubit_gates + analysis.multi_qubit_gates)
        )
        return {
            "communication_semantics": "amplitude_shard_transport",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "estimated_communication_bytes": int(memory)
            * max(1, distributed_gate_count),
            "collective_counts": {
                "all_to_all_or_pair_exchange": distributed_gate_count
            },
            "p2p_counts": {"pair_exchange": distributed_gate_count},
            "communication_frequency": "per_sharded_wire_gate",
            "rank_ownership": rank_ownership,
            "topology_dependency": "collective_route_depends_on_runtime_backend",
        }
    if state == "mps":
        boundary_count = max(0, world_size - 1)
        boundary_bytes = boundary_count * max(1, int(complex_bytes)) * 4
        return {
            "communication_semantics": "mps_site_or_bond_boundary_exchange",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "estimated_communication_bytes": boundary_bytes * max(1, analysis.depth),
            "collective_counts": {},
            "p2p_counts": {
                "boundary_exchange": boundary_count * max(1, analysis.depth)
            },
            "communication_frequency": "per_boundary_gate_or_canonicalization_step",
            "rank_ownership": rank_ownership,
            "topology_dependency": "rank_layout_affects_boundary_cost",
        }
    reduce_count = max(1, analysis.n_instructions)
    return {
        "communication_semantics": "tensor_network_slice_reduce",
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": node_count,
        "estimated_communication_bytes": int(complex_bytes) * world_size * reduce_count,
        "collective_counts": {"all_reduce_or_reduce_scatter": reduce_count},
        "p2p_counts": {},
        "communication_frequency": "per_slice_batch_or_observable",
        "rank_ownership": rank_ownership,
        "topology_dependency": "collective_route_depends_on_runtime_backend",
    }


def gradient_plan(
    *,
    mode: str,
    gradient: str,
    distribution_semantics: str,
    require_gradients: bool,
    blockers: tuple[str, ...],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ready = gradient in {
        "native_autograd",
        "jax_value_and_grad",
        "jax_sharded_parameter_vjp",
        "not_required",
    }
    parameter_gradient_ready = bool(
        ready and not blockers and (not require_gradients or gradient != "not_required")
    )
    return {
        "gradient_support": gradient,
        "required": bool(require_gradients),
        "parameter_gradient_ready": parameter_gradient_ready,
        "gradient_distribution_semantics": (
            distribution_semantics if ready else "incomplete"
        ),
        "optimizer_update_semantics": (
            "local_optimizer_step"
            if distribution_semantics == "single_device_fast_path"
            else (
                "requires_distributed_gradient_reduction"
                if distribution_semantics == "sharded_across_ranks"
                else "not_a_capacity_scaling_training_path"
            )
        ),
        "checkpointing_strategy": "runtime_specific",
        "blockers": blockers,
        "fail_closed": bool(blockers),
        "mode": mode,
        "details": dict(details or {}),
    }


def deployment_plan(
    *,
    mode: str,
    state: str,
    deployment_ready: bool,
    require_deployment: bool,
    blockers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "deployment_ready": bool(deployment_ready),
        "required": bool(require_deployment),
        "parameter_freeze_required": True,
        "source_ir": "FlagQuantum IR",
        "state_mode": state,
        "training_runtime_mode": mode,
        "next_steps": (
            "freeze_parameters",
            "lower_to_target_gate_set",
            "route_and_optimize",
            "submit_or_export",
        ),
        "blockers": blockers if require_deployment and not deployment_ready else (),
    }


__all__ = [
    "CircuitAnalysisView",
    "communication_plan",
    "deployment_plan",
    "gradient_plan",
    "memory_plan",
]
