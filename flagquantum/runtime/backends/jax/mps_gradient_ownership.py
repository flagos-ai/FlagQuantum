"""MPS parameter-gradient ownership, transfers, and observables."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ....simulation.jax_mps import jax_sharded_mps_z_sum
from .mps_training_records import JAXShardedMPSParameterFlowPlan
from .runtime_environment import (
    _jax_complex_dtype,
    _require_jax,
)


def _execute_local_mps_parameter_gradient_ownership(
    circuit_builder: Callable[[Any], Any],
    parameter_array: Any,
    gradient: Any,
    parameter_flow_plan: JAXShardedMPSParameterFlowPlan,
    shard_plans: Sequence[Any],
) -> dict[str, Any]:
    """Attribute a local reference gradient to the existing MPS owner plan."""

    import numpy as np

    jax, jnp = _require_jax()
    from .kernel import _JAXParameterProxy

    assignments = tuple(parameter_flow_plan.assignments)
    occurrences = tuple(
        (assignment, str(parameter_name))
        for assignment in assignments
        for parameter_name in assignment.parameter_names
    )
    world_size = int(parameter_flow_plan.world_size)
    rank_site_ranges = {
        int(shard.rank): tuple(int(wire) for wire in shard.wires)
        for shard in shard_plans
    }
    gradient_host = np.asarray(gradient)
    flat_gradient = gradient_host.reshape(-1)
    parameter_count = int(flat_gradient.size)
    records: list[dict[str, Any]] = []
    execution_blockers: list[str] = []
    claim_blockers = [
        "mps_parameter_gradient_ownership_local_simulation_only",
        "mps_production_rank_local_parameter_vjp_pending",
    ]

    if world_size <= 1:
        execution_blockers.append("mps_parameter_gradient_world_size_must_exceed_one")
    if not occurrences:
        execution_blockers.append("mps_parameter_gradient_occurrences_missing")

    def _gate_parameter_vector(params: Any) -> Any:
        circuit = circuit_builder(_JAXParameterProxy(params))
        instructions = tuple(circuit.to_ir().instructions)
        values = []
        for assignment, parameter_name in occurrences:
            instruction_index = int(assignment.instruction_index)
            if instruction_index >= len(instructions):
                raise ValueError("mps_parameter_gradient_instruction_index_missing")
            instruction = instructions[instruction_index]
            if str(instruction.name) != str(assignment.name):
                raise ValueError("mps_parameter_gradient_gate_identity_mismatch")
            if parameter_name not in instruction.params:
                raise ValueError("mps_parameter_gradient_gate_parameter_missing")
            value = jnp.asarray(instruction.params[parameter_name], dtype=params.dtype)
            if value.ndim != 0:
                if int(value.size) != 1:
                    raise ValueError("mps_parameter_gradient_gate_parameter_not_scalar")
                value = value.reshape(())
            values.append(value)
        return jnp.stack(values) if values else jnp.empty((0,), dtype=params.dtype)

    dependency_rows: list[set[int]] = [set() for _ in occurrences]
    try:
        parameter_array = jnp.asarray(parameter_array)
        probe = jnp.arange(parameter_array.size, dtype=parameter_array.dtype).reshape(
            parameter_array.shape
        ) * jnp.asarray(0.173, dtype=parameter_array.dtype) + jnp.asarray(
            0.311, dtype=parameter_array.dtype
        )
        for dependency_input in (parameter_array, probe):
            jacobian = np.asarray(jax.jacrev(_gate_parameter_vector)(dependency_input))
            jacobian = jacobian.reshape((len(occurrences), -1))
            tolerance = np.finfo(jacobian.dtype).eps * 32
            for row_index, row in enumerate(jacobian):
                dependency_rows[row_index].update(
                    int(index) for index in np.flatnonzero(np.abs(row) > tolerance)
                )
    except Exception as exc:
        execution_blockers.append(
            f"mps_parameter_gradient_dependency_mapping_failed:{type(exc).__name__}"
        )

    routes_by_parameter: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(parameter_count)
    }
    for occurrence_index, (assignment, parameter_name) in enumerate(occurrences):
        dependencies = (
            dependency_rows[occurrence_index]
            if occurrence_index < len(dependency_rows)
            else set()
        )
        gate_id = f"instruction_{int(assignment.instruction_index)}"
        if len(dependencies) != 1:
            blocker = (
                f"mps_parameter_gradient_dependency_missing:{gate_id}:{parameter_name}"
                if not dependencies
                else f"mps_parameter_gradient_dependency_ambiguous:{gate_id}:{parameter_name}"
            )
            execution_blockers.append(blocker)
            continue
        parameter_flat_index = next(iter(dependencies))
        if not 0 <= parameter_flat_index < parameter_count:
            execution_blockers.append(
                f"mps_parameter_gradient_index_out_of_range:{gate_id}:{parameter_name}"
            )
            continue
        owner_rank = assignment.owner_rank
        if owner_rank is None or not 0 <= int(owner_rank) < world_size:
            execution_blockers.append(
                f"mps_parameter_gradient_owner_missing:{gate_id}:{parameter_name}"
            )
            continue
        parameter_index = tuple(
            int(index)
            for index in np.unravel_index(
                parameter_flat_index, tuple(int(dim) for dim in gradient_host.shape)
            )
        )
        parameter_id = (
            "parameter[" + ",".join(str(index) for index in parameter_index) + "]"
        )
        crosses_boundary = assignment.locality == "boundary"
        record_blockers: tuple[str, ...] = ()
        if crosses_boundary:
            route_blocker = f"mps_boundary_parameter_gradient_requires_executed_adjoint_route:{gate_id}"
            record_blockers = (route_blocker,)
            claim_blockers.append(route_blocker)
        route = {
            "parameter_id": parameter_id,
            "parameter_flat_index": parameter_flat_index,
            "parameter_index": parameter_index,
            "gate_id": gate_id,
            "instruction_index": int(assignment.instruction_index),
            "gate_name": str(assignment.name),
            "parameter_name": parameter_name,
            "owner_rank": int(owner_rank),
            "site_range": rank_site_ranges.get(int(owner_rank), ()),
            "touched_ranks": tuple(int(rank) for rank in assignment.touched_ranks),
            "crosses_shard_boundary": crosses_boundary,
            "gradient_distribution_semantics": "rank_owned_attribution",
            "parameter_gradient_ownership": {
                "owner_rank": int(owner_rank),
                "ownership_semantics": "rank_owned_parameter_gradient",
            },
            "gradient_route": str(assignment.gradient_route),
            "gradient_value": float(flat_gradient[parameter_flat_index]),
            "gradient_dtype": str(gradient_host.dtype),
            "gradient_bytes": int(gradient_host.dtype.itemsize),
            "blockers": record_blockers,
        }
        records.append(route)
        routes_by_parameter[parameter_flat_index].append(route)

    ownership: list[dict[str, Any]] = []
    rank_parameter_indices: dict[int, list[int]] = {
        rank: [] for rank in range(max(1, world_size))
    }
    for parameter_flat_index in range(parameter_count):
        routes = routes_by_parameter.get(parameter_flat_index, ())
        if not routes:
            execution_blockers.append(
                f"mps_parameter_gradient_owner_missing:parameter_{parameter_flat_index}"
            )
            continue
        owners = {int(route["owner_rank"]) for route in routes}
        if len(owners) != 1:
            execution_blockers.append(
                f"mps_shared_parameter_cross_rank_reduction_pending:parameter_{parameter_flat_index}"
            )
            continue
        owner_rank = next(iter(owners))
        parameter_index = tuple(
            int(index)
            for index in np.unravel_index(
                parameter_flat_index, tuple(int(dim) for dim in gradient_host.shape)
            )
        )
        parameter_id = (
            "parameter[" + ",".join(str(index) for index in parameter_index) + "]"
        )
        ownership.append(
            {
                "parameter_id": parameter_id,
                "parameter_flat_index": parameter_flat_index,
                "parameter_index": parameter_index,
                "owner_rank": owner_rank,
                "site_range": rank_site_ranges.get(owner_rank, ()),
                "gate_ids": tuple(
                    dict.fromkeys(str(route["gate_id"]) for route in routes)
                ),
                "crosses_shard_boundary": any(
                    bool(route["crosses_shard_boundary"]) for route in routes
                ),
                "gradient_routes": tuple(
                    dict.fromkeys(str(route["gradient_route"]) for route in routes)
                ),
                "ownership_semantics": "rank_owned_parameter_gradient",
                "gradient_value": float(flat_gradient[parameter_flat_index]),
                "gradient_dtype": str(gradient_host.dtype),
                "gradient_bytes": int(gradient_host.dtype.itemsize),
            }
        )
        rank_parameter_indices[owner_rank].append(parameter_flat_index)

    rank_gradients = []
    local_gradient_bytes_by_rank = []
    for rank in range(max(1, world_size)):
        indices = tuple(rank_parameter_indices.get(rank, ()))
        local_gradient_bytes = int(len(indices) * gradient_host.dtype.itemsize)
        local_gradient_bytes_by_rank.append(local_gradient_bytes)
        rank_gradients.append(
            {
                "rank": rank,
                "site_range": rank_site_ranges.get(rank, ()),
                "parameter_ids": tuple(
                    item["parameter_id"]
                    for item in ownership
                    if int(item["owner_rank"]) == rank
                ),
                "parameter_flat_indices": indices,
                "gradient_values": tuple(
                    float(flat_gradient[index]) for index in indices
                ),
                "local_gradient_bytes": local_gradient_bytes,
                "ownership_semantics": "rank_owned_parameter_gradient_slice",
            }
        )

    execution_blockers_out = tuple(dict.fromkeys(execution_blockers))
    blockers = tuple(dict.fromkeys((*execution_blockers_out, *claim_blockers)))
    return {
        "status": "blocked" if execution_blockers_out else "local_cpu_attributed",
        "valid": not execution_blockers_out,
        "execution_scope": "development_cpu",
        "evidence_scope": "parameter_gradient_ownership_attribution",
        "claim_evidence_type": "development_smoke",
        "gradient_source": "local_simulated_global_parameter_gradient",
        "gradient_distribution_semantics": (
            "local_simulation_with_rank_owned_attribution"
            if world_size > 1
            else "replicated_single_rank"
        ),
        "parameter_gradient_ownership_semantics": "rank_owned_attribution",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "world_size": world_size,
        "parameter_count": parameter_count,
        "owned_parameter_count": len(ownership),
        "dependency_probe_count": 2,
        "parameter_gradient_ownership": tuple(ownership),
        "rank_owned_gradients": tuple(rank_gradients),
        "local_gradient_bytes_by_rank": tuple(local_gradient_bytes_by_rank),
        "records": tuple(records),
        "replicated_rank_autograd": False,
        "full_local_mps_replay": False,
        "statevector_fallback": False,
        "execution_blockers": execution_blockers_out,
        "blockers": blockers,
    }


def _jax_sharded_mps_z_sum_from_rank_tensors(
    rank_tensors: Mapping[int, Mapping[int, Any]],
    *,
    n_wires: int,
    bsz: int,
    complex_bytes: int,
    observable_wires: Sequence[int] | None,
) -> Any:
    return jax_sharded_mps_z_sum(
        rank_tensors,
        n_wires=int(n_wires),
        batch_size=int(bsz),
        dtype=_jax_complex_dtype(complex_bytes),
        observable_wires=observable_wires,
    )
