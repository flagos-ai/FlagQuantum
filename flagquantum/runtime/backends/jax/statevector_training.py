"""Statevector training device checks and training-plan construction."""

from __future__ import annotations

from typing import Any

from ...distributed.backend_policy import DistributedBackendPolicy
from .backend_dispatch import plan_jax_distributed_quantum_backend
from .planning_core import _as_ir
from .runtime_environment import (
    _jax_device_count_summary,
    _resolve_jax_backward_backend,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
)
from .statevector.kernels import (
    _statevector_pmap_backward_blockers,
    _statevector_shard_map_backward_blockers,
)
from .statevector.records import JAXShardedStatevectorTrainingPlan


def _statevector_training_device_blockers(
    *,
    backend: str,
    world_size: int,
    inspect_devices: bool,
    assume_devices_ready: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if backend == "local_simulated":
        return {}, ("local_simulated_backward_not_capacity_scaling",)
    if not inspect_devices:
        if assume_devices_ready:
            return {"assumed_ready": True}, ()
        return {}, ("production_device_preflight_required",)

    device_summary = _jax_device_count_summary()
    blockers: list[str] = []
    if int(device_summary["global_device_count"]) < int(world_size):
        blockers.append("insufficient_global_jax_devices")
    if backend == "pmap":
        if int(device_summary["local_device_count"]) <= 0:
            blockers.append("pmap_requires_one_local_device_per_process")
    elif backend == "shard_map":
        if int(device_summary["process_count"]) > 1:
            blockers.append("shard_map_multiprocess_global_array_input_pending")
        if int(device_summary["local_device_count"]) < int(world_size):
            blockers.append("shard_map_requires_world_size_local_devices")
    else:
        blockers.append(f"unsupported_statevector_backward_backend:{backend}")
    return device_summary, tuple(blockers)


def plan_jax_sharded_statevector_training(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int = 8,
    backward_backend: str = "auto",
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    inspect_devices: bool = False,
    assume_devices_ready: bool = False,
) -> JAXShardedStatevectorTrainingPlan:
    """Plan whether JAX sharded statevector backward is a claimable path.

    The function is intentionally a preflight planner: it does not execute a
    quantum kernel and it never turns local simulation into a scalability claim.
    """

    from ..statevector.planning import plan_distributed_statevector

    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    resolved_backward_backend = _resolve_jax_backward_backend(backward_backend, policy)
    ir = _as_ir(circuit_or_ir)
    state_plan = plan_distributed_statevector(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    if resolved_backward_backend == "pmap":
        static_blockers = _statevector_pmap_backward_blockers(state_plan)
        backward_execution = "jax_pmap_backward"
    elif resolved_backward_backend == "shard_map":
        static_blockers = _statevector_shard_map_backward_blockers(state_plan)
        backward_execution = "jax_shard_map_backward"
    else:
        static_blockers = ()
        backward_execution = "local_simulated_backward"

    device_summary, device_blockers = _statevector_training_device_blockers(
        backend=resolved_backward_backend,
        world_size=resolved_world_size,
        inspect_devices=inspect_devices,
        assume_devices_ready=assume_devices_ready,
    )
    blockers = tuple(dict.fromkeys((*static_blockers, *device_blockers)))
    is_sharded = (
        int(state_plan.world_size) > 1
        and str(state_plan.distribution) != "replicated_single_rank"
    )
    gradient_ready = bool(
        is_sharded
        and resolved_backward_backend in {"pmap", "shard_map"}
        and not blockers
    )
    claimable = bool(
        gradient_ready
        and policy.profile == "production"
        and (inspect_devices or assume_devices_ready)
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="statevector",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        distributed_backend_policy=policy,
    )
    return JAXShardedStatevectorTrainingPlan(
        mode="statevector",
        backend=resolved_backward_backend,
        backward_execution=backward_execution,
        distribution_semantics=(
            "sharded_across_ranks" if is_sharded else "replicated_single_rank"
        ),
        scalability_claim_allowed=claimable,
        gradient_ready=gradient_ready,
        world_size=int(state_plan.world_size),
        local_world_size=int(state_plan.local_world_size),
        node_count=int(state_plan.node_count),
        n_wires=int(state_plan.n_wires),
        batch_size=int(state_plan.bsz),
        static_blockers=tuple(static_blockers),
        device_blockers=tuple(device_blockers),
        blockers=blockers,
        statevector_plan_summary=state_plan.summary(),
        jax_plan_summary=jax_plan.summary(),
        device_summary=device_summary,
        inspected_devices=bool(inspect_devices),
    )
