# ruff: noqa: F401, F821
"""MPS parameter gradients, backend blockers, and parameter-flow planning."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .common import communication_tier as _communication_tier
from .common import env_int as _env_int
from .common import node_count as _node_count
from .common import product_int as _product
from .common import rank_for_wire as _rank_for_wire
from .common import split_contiguous as _split_contiguous
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)


def jax_sharded_mps_parameter_value_and_grad(
    circuit_builder: Callable[[Any], Any],
    parameters: Any,
    *,
    n_wires: int,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    observable: str = "z_sum",
    observable_wires: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    backward_backend: str = "auto",
    jit: bool = False,
) -> JAXShardedMPSParameterGradientResult:
    """Reverse-mode value/gradient for a site-sharded parameterized MPS circuit."""

    from ...distributed.engine import _instruction_is_boundary_local, _mps_shards

    torch = _require_torch()
    jax, jnp = _require_jax()
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
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    static_parameters = _torch_parameters_for_static_build(
        parameters, complex_bytes=complex_bytes
    )
    parameter_shape = tuple(int(dim) for dim in static_parameters.shape)
    example_circuit = circuit_builder(static_parameters)
    ir = _as_ir(example_circuit)
    if int(ir.n_wires) != int(n_wires):
        raise ValueError(
            f"n_wires={n_wires} does not match circuit IR n_wires={ir.n_wires}."
        )
    shard_plans = _mps_shards(ir.n_wires, resolved_world_size)
    parameter_flow_plan = plan_jax_sharded_mps_parameter_flow(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        distributed_backend_policy=policy,
    )
    boundary_preflight_count = sum(
        1
        for instruction in ir
        if _instruction_is_boundary_local(instruction, shard_plans)
    )
    resolved_backward_backend = _resolve_jax_backward_backend(backward_backend, policy)
    production_preflight = _require_jax_production_backward_ready(
        mode="mps",
        backend=resolved_backward_backend,
        world_size=resolved_world_size,
        blockers=(
            _mps_pmap_backward_blockers(
                world_size=resolved_world_size,
                boundary_sync_count=boundary_preflight_count,
            )
            if resolved_backward_backend != "local_simulated"
            else ()
        ),
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="mps",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
        distributed_backend_policy=policy,
    )
    jax_device = _resolve_jax_device(device)
    jax_parameters = _jax_parameter_array_from_input(
        parameters, complex_bytes=complex_bytes
    )
    compute_dtype = "complex128" if int(complex_bytes) == 16 else "complex64"

    def _loss(parameter_array: Any) -> Any:
        from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

        previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
        try:
            parameter_array = jnp.asarray(
                parameter_array, dtype=_jax_real_dtype(complex_bytes)
            )
            circuit = circuit_builder(_JAXParameterProxy(parameter_array))
            rank_tensors, *_ = _jax_parameterized_mps_rank_tensors(
                circuit,
                n_wires=int(n_wires),
                bsz=bsz,
                shard_plans=shard_plans,
                complex_bytes=complex_bytes,
                max_bond=max_bond,
                cutoff=cutoff,
                local_world_size=resolved_local_world_size,
                device=jax_device,
            )
            if str(observable) not in {"z", "z_sum"}:
                raise ValueError(
                    "JAX sharded MPS parameter gradients currently support observable='z_sum' or 'z'."
                )
            return _jax_sharded_mps_z_sum_from_rank_tensors(
                rank_tensors,
                n_wires=int(n_wires),
                bsz=bsz,
                complex_bytes=complex_bytes,
                observable_wires=observable_wires,
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)

    value_and_grad = jax.value_and_grad(_loss)
    if jit:
        value_and_grad = jax.jit(value_and_grad)
    value, gradient = value_and_grad(jax_parameters)

    from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
    try:
        summary_circuit = circuit_builder(_JAXParameterProxy(jax_parameters))
        (
            rank_tensors,
            owned_instruction_count,
            sharded_kernel_count,
            boundary_sync_count,
            boundary_transfer_bytes,
            truncation_records,
            boundary_protocols,
        ) = _jax_parameterized_mps_rank_tensors(
            summary_circuit,
            n_wires=int(n_wires),
            bsz=bsz,
            shard_plans=shard_plans,
            complex_bytes=complex_bytes,
            max_bond=max_bond,
            cutoff=cutoff,
            local_world_size=resolved_local_world_size,
            device=jax_device,
        )
    finally:
        _set_active_jax_compute_dtype(previous_dtype)

    boundary_adjoint_exchange_evidence = _execute_local_mps_boundary_adjoint_exchange(
        rank_tensors,
        boundary_protocols,
        shard_plans,
    )
    parameter_gradient_ownership_evidence = (
        _execute_local_mps_parameter_gradient_ownership(
            circuit_builder,
            jax_parameters,
            gradient,
            parameter_flow_plan,
            shard_plans,
        )
    )

    return JAXShardedMPSParameterGradientResult(
        value=value,
        gradient=gradient,
        rank_shards=_rank_shards_from_jax_mps_tensors(rank_tensors, shard_plans),
        shard_plans=tuple(shard_plans),
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=ir.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        parameter_shape=parameter_shape,
        max_bond=max_bond,
        cutoff=float(cutoff),
        owned_instruction_count=owned_instruction_count,
        sharded_kernel_count=sharded_kernel_count,
        boundary_sync_count=boundary_sync_count,
        boundary_transfer_bytes=boundary_transfer_bytes,
        truncation_records=tuple(truncation_records),
        boundary_protocols=tuple(boundary_protocols),
        boundary_adjoint_exchange_evidence=boundary_adjoint_exchange_evidence,
        parameter_gradient_ownership_evidence=parameter_gradient_ownership_evidence,
        observable=str(observable),
        backward_backend=resolved_backward_backend,
        backward_execution=str(production_preflight["execution"]),
        production_device_summary=production_preflight["device_summary"],
        production_blockers=tuple(production_preflight["blockers"]),
        parameters=jax_parameters,
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
