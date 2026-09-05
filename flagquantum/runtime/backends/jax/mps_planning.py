"""Sharded MPS training and representation planning."""

from __future__ import annotations

from typing import Any

from ....core.ir import CircuitIR
from ...distributed.backend_policy import DistributedBackendPolicy
from .common import communication_tier as _communication_tier
from .common import node_count as _node_count
from .common import rank_for_wire as _rank_for_wire
from .common import split_contiguous as _split_contiguous
from .mps_training_records import (
    JAXShardedMPSParameterFlowPlan,
    JAXShardedMPSParameterGateAssignment,
    JAXShardedMPSTrainingPlan,
)
from .planning_core import JAXDistributedQuantumPlan, _as_ir
from .runtime_environment import (
    _jax_device_count_summary,
    _resolve_jax_backward_backend,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
)


def _mps_pmap_backward_blockers(
    *, world_size: int, boundary_sync_count: int
) -> tuple[str, ...]:
    blockers: list[str] = []
    if int(world_size) <= 1:
        blockers.append("world_size_is_one")
    blockers.append("pmap_mps_rank_environment_scan_pending")
    if int(boundary_sync_count) > 0:
        blockers.append("pmap_mps_boundary_tensor_transport_pending")
    return tuple(blockers)


def _mps_shard_map_backward_blockers(
    *, world_size: int, boundary_sync_count: int
) -> tuple[str, ...]:
    blockers: list[str] = []
    if int(world_size) <= 1:
        blockers.append("world_size_is_one")
    blockers.extend(
        (
            "shard_map_mps_global_array_layout_pending",
            "shard_map_mps_rank_environment_scan_pending",
        )
    )
    if int(boundary_sync_count) > 0:
        blockers.append("shard_map_mps_boundary_tensor_transport_pending")
    return tuple(blockers)


def _mps_training_device_blockers(
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
            blockers.append("shard_map_mps_multiprocess_global_array_input_pending")
        if int(device_summary["local_device_count"]) < int(world_size):
            blockers.append("shard_map_requires_world_size_local_devices")
    else:
        blockers.append(f"unsupported_mps_backward_backend:{backend}")
    return device_summary, tuple(blockers)


def plan_jax_sharded_mps_parameter_flow(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
) -> JAXShardedMPSParameterFlowPlan:
    """Plan parameter ownership and gradient routes for sharded MPS backward."""

    from ...distributed.engine import (
        _boundary_sync_record,
        _instruction_is_boundary_local,
        _instruction_is_site_local,
        _instruction_owner,
        _mps_shards,
    )
    from ...distributed.engine import (
        _rank_for_wire as _distributed_rank_for_wire,
    )

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
    ir = _as_ir(circuit_or_ir)
    shard_plans = _mps_shards(ir.n_wires, resolved_world_size)
    rank_counts = [0 for _ in range(int(resolved_world_size))]
    assignments: list[JAXShardedMPSParameterGateAssignment] = []
    boundary_edges: list[dict[str, Any]] = []
    blockers: list[str] = []
    if int(resolved_world_size) <= 1:
        blockers.append("world_size_is_one")

    for index, instruction in enumerate(ir):
        parameter_names = tuple(str(name) for name in sorted(instruction.params))
        if not parameter_names:
            continue
        wires = tuple(int(wire) for wire in instruction.wires)
        touched_ranks = tuple(
            sorted(
                {int(_distributed_rank_for_wire(wire, shard_plans)) for wire in wires}
            )
        )
        site_local = _instruction_is_site_local(instruction, shard_plans)
        boundary_local = _instruction_is_boundary_local(instruction, shard_plans)
        if site_local:
            owner = int(_instruction_owner(instruction, shard_plans))
            rank_counts[owner] += len(parameter_names)
            assignments.append(
                JAXShardedMPSParameterGateAssignment(
                    instruction_index=index,
                    name=str(instruction.name),
                    wires=wires,
                    parameter_names=parameter_names,
                    owner_rank=owner,
                    touched_ranks=touched_ranks,
                    locality="site_local",
                    gradient_route="rank_local_parameter_vjp",
                    communication_tier="none",
                )
            )
            continue
        if boundary_local:
            boundary = _boundary_sync_record(instruction, shard_plans)
            owner = int(boundary.owner_rank)
            rank_counts[owner] += len(parameter_names)
            tier = _communication_tier(
                boundary.left_rank,
                boundary.right_rank,
                local_world_size=resolved_local_world_size,
            )
            edge = {
                "instruction_index": index,
                "name": str(instruction.name),
                "parameter_names": parameter_names,
                "left_wire": int(boundary.left_wire),
                "right_wire": int(boundary.right_wire),
                "left_rank": int(boundary.left_rank),
                "right_rank": int(boundary.right_rank),
                "owner_rank": owner,
                "tier": tier,
                "required_backward_transport": "boundary_adjoint_tensor_exchange",
            }
            boundary_edges.append(edge)
            assignments.append(
                JAXShardedMPSParameterGateAssignment(
                    instruction_index=index,
                    name=str(instruction.name),
                    wires=wires,
                    parameter_names=parameter_names,
                    owner_rank=owner,
                    touched_ranks=touched_ranks,
                    locality="boundary",
                    gradient_route="boundary_parameter_vjp",
                    communication_tier=tier,
                    blockers=(
                        "mps_boundary_parameter_pullback_pending",
                        "mps_boundary_parameter_gradient_requires_executed_adjoint_route",
                    ),
                )
            )
            blockers.extend(
                (
                    "mps_boundary_parameter_pullback_pending",
                    "mps_boundary_parameter_gradient_requires_executed_adjoint_route",
                )
            )
            continue

        assignments.append(
            JAXShardedMPSParameterGateAssignment(
                instruction_index=index,
                name=str(instruction.name),
                wires=wires,
                parameter_names=parameter_names,
                owner_rank=None,
                touched_ranks=touched_ranks,
                locality="unsupported",
                gradient_route="unsupported_without_full_mps_replay",
                communication_tier="unsupported",
                blockers=("mps_parameter_flow_unsupported_nonlocal_gate",),
            )
        )
        blockers.append("mps_parameter_flow_unsupported_nonlocal_gate")

    return JAXShardedMPSParameterFlowPlan(
        world_size=int(resolved_world_size),
        local_world_size=int(resolved_local_world_size),
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        n_wires=int(ir.n_wires),
        site_shard_ownership=tuple(
            {
                "rank": int(shard.rank),
                "wires": tuple(int(wire) for wire in shard.wires),
                "ownership_semantics": "mps_site_range",
            }
            for shard in shard_plans
        ),
        bond_shard_ownership=tuple(
            {
                "left_rank": int(shard.rank),
                "right_rank": int(shard.rank + 1),
                "left_wire": int(shard.right_boundary),
                "right_wire": int(shard.right_boundary + 1),
                "ownership_semantics": "adjacent_rank_boundary_bond",
            }
            for shard in shard_plans
            if shard.right_boundary is not None
            and int(shard.rank + 1) < int(resolved_world_size)
        ),
        assignments=tuple(assignments),
        rank_parameter_counts=tuple(int(count) for count in rank_counts),
        boundary_parameter_edges=tuple(boundary_edges),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def plan_jax_sharded_mps_training(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int = 8,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    backward_backend: str = "auto",
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    inspect_devices: bool = False,
    assume_devices_ready: bool = False,
) -> JAXShardedMPSTrainingPlan:
    """Plan whether JAX site-sharded MPS backward is a production training path.

    The current production MPS backward intentionally fails closed.  This
    planner records shard ownership, boundary communication, and the exact
    blockers so user-facing runtime selection can stay honest while the
    executor is being completed.
    """

    from ...distributed.engine import _instruction_is_boundary_local, _mps_shards

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
    shard_plans = _mps_shards(ir.n_wires, resolved_world_size)
    parameter_flow = plan_jax_sharded_mps_parameter_flow(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        distributed_backend_policy=policy,
    )
    parameter_flow_summary = parameter_flow.summary()
    parameter_flow_blockers = tuple(
        str(item) for item in parameter_flow_summary.get("blockers", ())
    )
    boundary_sync_count = sum(
        1
        for instruction in ir
        if _instruction_is_boundary_local(instruction, shard_plans)
    )
    if resolved_backward_backend == "pmap":
        backend_static_blockers = _mps_pmap_backward_blockers(
            world_size=resolved_world_size,
            boundary_sync_count=boundary_sync_count,
        )
        backward_execution = "jax_pmap_backward"
    elif resolved_backward_backend == "shard_map":
        backend_static_blockers = _mps_shard_map_backward_blockers(
            world_size=resolved_world_size,
            boundary_sync_count=boundary_sync_count,
        )
        backward_execution = "jax_shard_map_backward"
    else:
        backend_static_blockers = ()
        backward_execution = "local_simulated_backward"

    static_blockers = tuple(
        dict.fromkeys(
            (
                "mps_boundary_adjoint_exchange_pending",
                "mps_parameter_gradient_ownership_pending",
                "mps_optimizer_update_ownership_pending",
                "mps_canonicalization_backward_strategy_pending",
                *(
                    ("mps_truncation_gradient_metadata_incomplete",)
                    if float(cutoff) > 0.0
                    else ()
                ),
                *backend_static_blockers,
                *parameter_flow_blockers,
            )
        )
    )
    device_summary, device_blockers = _mps_training_device_blockers(
        backend=resolved_backward_backend,
        world_size=resolved_world_size,
        inspect_devices=inspect_devices,
        assume_devices_ready=assume_devices_ready,
    )
    blockers = tuple(dict.fromkeys((*static_blockers, *device_blockers)))
    jax_plan = _mps_plan(
        ir,
        policy=policy,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
    )
    jax_summary = jax_plan.summary()
    is_sharded = int(resolved_world_size) > 1
    unsupported_parameter_gate_count = int(
        parameter_flow_summary.get("unsupported_parameter_gate_count", 0)
    )
    local_development_gradient_ready = bool(
        is_sharded
        and resolved_backward_backend == "local_simulated"
        and unsupported_parameter_gate_count == 0
    )
    return JAXShardedMPSTrainingPlan(
        mode="mps",
        backend=resolved_backward_backend,
        backward_execution=backward_execution,
        distribution_semantics=(
            str(jax_summary.get("distribution_semantics", "requires_runtime_summary"))
            if is_sharded
            else "replicated_single_rank"
        ),
        intended_distribution_semantics=str(
            jax_summary.get("intended_distribution_semantics", "sharded_across_ranks")
        ),
        scalability_claim_allowed=False,
        gradient_ready=False,
        local_development_gradient_ready=local_development_gradient_ready,
        world_size=int(resolved_world_size),
        local_world_size=int(resolved_local_world_size),
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        n_wires=int(ir.n_wires),
        batch_size=int(bsz),
        max_bond=max_bond,
        cutoff=float(cutoff),
        boundary_sync_count=int(boundary_sync_count),
        static_blockers=static_blockers,
        device_blockers=tuple(device_blockers),
        blockers=blockers,
        jax_plan_summary=jax_summary,
        parameter_flow_summary=parameter_flow_summary,
        device_summary=device_summary,
        inspected_devices=bool(inspect_devices),
    )


def _mps_plan(
    ir: CircuitIR,
    *,
    policy: DistributedBackendPolicy,
    world_size: int,
    local_world_size: int,
    bsz: int,
    complex_bytes: int,
    max_bond: int | None,
) -> JAXDistributedQuantumPlan:
    wire_shards = _split_contiguous(ir.n_wires, world_size)
    bond = max(1, int(max_bond or 2))
    rank_ownership = []
    local_memory = []
    for rank, wires in enumerate(wire_shards):
        # Conservative MPS tensor estimate: each owned site has two physical
        # states and up to chi^2 virtual entries per batch item.
        memory = len(wires) * int(bsz) * 2 * bond * bond * int(complex_bytes)
        local_memory.append(int(memory))
        rank_ownership.append(
            {
                "rank": rank,
                "state_partition": "mps_site_range",
                "wires": tuple(int(wire) for wire in wires),
                "local_memory_bytes": int(memory),
            }
        )
    boundary_edges = []
    intra_bytes = 0
    inter_bytes = 0
    tensor_bytes = int(bsz) * 2 * bond * bond * int(complex_bytes)
    for left_wire in range(max(0, ir.n_wires - 1)):
        left_rank = _rank_for_wire(left_wire, wire_shards)
        right_rank = _rank_for_wire(left_wire + 1, wire_shards)
        if left_rank == right_rank:
            continue
        transfer = tensor_bytes * 4
        tier = _communication_tier(
            left_rank, right_rank, local_world_size=local_world_size
        )
        if tier == "intra_node":
            intra_bytes += transfer
        else:
            inter_bytes += transfer
        boundary_edges.append(
            {
                "left_wire": left_wire,
                "right_wire": left_wire + 1,
                "left_rank": left_rank,
                "right_rank": right_rank,
                "tier": tier,
                "estimated_transfer_bytes": transfer,
            }
        )
    blockers = (
        "jax_pmap_mps_site_sharded_executor_pending",
        "rank_local_jax_kernel_is_not_capacity_scaling",
    )
    gradient_blockers = (
        "mps_boundary_adjoint_exchange_pending",
        "mps_parameter_gradient_ownership_pending",
        "mps_optimizer_update_ownership_pending",
    )
    return JAXDistributedQuantumPlan(
        mode="mps",
        n_wires=ir.n_wires,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=_node_count(world_size, local_world_size),
        backend_policy=policy,
        rank_ownership=tuple(rank_ownership),
        communication_tiers={
            "model": "jax_pmap_mps_boundary_exchange_planned",
            "boundary_edge_count": len(boundary_edges),
            "boundary_edges": tuple(boundary_edges),
            "estimated_transfer_bytes": int(intra_bytes + inter_bytes),
            "intra_node_communication_bytes": int(intra_bytes),
            "inter_node_communication_bytes": int(inter_bytes),
        },
        local_memory_bytes_by_rank=tuple(local_memory),
        blockers=blockers,
        gradient_blockers=gradient_blockers,
        task_summary={
            "partition": "contiguous_mps_sites",
            "max_bond": max_bond,
            "estimated_tensor_bytes": tensor_bytes,
        },
    )
