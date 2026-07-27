# ruff: noqa: F401, F821
"""MPS tensor initialization, SVD splitting, reconstruction, and resources."""

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


def _initialize_jax_mps_rank_tensors(
    *,
    n_wires: int,
    bsz: int,
    shard_plans: Sequence[Any],
    dtype: Any,
    device: Any | None,
) -> dict[int, dict[int, Any]]:
    _, jnp = _require_jax()
    rank_tensors: dict[int, dict[int, Any]] = {}
    for shard in shard_plans:
        local: dict[int, Any] = {}
        for wire in shard.wires:
            tensor = jnp.zeros((int(bsz), 1, 2, 1), dtype=dtype)
            tensor = tensor.at[:, 0, 0, 0].set(jnp.asarray(1.0 + 0.0j, dtype=dtype))
            local[int(wire)] = _jnp_device_put(tensor, device)
        rank_tensors[int(shard.rank)] = local
    return rank_tensors


def _rank_shards_from_jax_mps_tensors(
    rank_tensors: Mapping[int, Mapping[int, Any]],
    shard_plans: Sequence[Any],
) -> tuple[JAXMPSRankShardState, ...]:
    return tuple(
        JAXMPSRankShardState(
            rank=int(shard.rank),
            wires=tuple(int(wire) for wire in shard.wires),
            local_tensors=dict(rank_tensors.get(int(shard.rank), {})),
        )
        for shard in shard_plans
    )


def _apply_one_jax_mps_tensor(tensor: Any, matrix: Any) -> Any:
    _, jnp = _require_jax()
    matrix = jnp.asarray(matrix, dtype=tensor.dtype)
    if matrix.ndim == 2:
        return jnp.einsum("pq,blqr->blpr", matrix, tensor, precision="highest")
    return jnp.einsum("bpq,blqr->blpr", matrix, tensor, precision="highest")


def _apply_two_jax_mps_tensors(
    left: Any,
    right: Any,
    matrix: Any,
    *,
    max_bond: int | None,
    cutoff: float,
    reverse: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    _, jnp = _require_jax()
    matrix = jnp.asarray(matrix, dtype=left.dtype)
    theta = jnp.einsum("blsm,bmtr->blstr", left, right, precision="highest")
    if reverse:
        theta = jnp.swapaxes(theta, 2, 3)
    theta = theta.reshape(left.shape[0], left.shape[1], 4, right.shape[3])
    if matrix.ndim == 2:
        theta = jnp.einsum("ij,bljr->blir", matrix, theta, precision="highest")
    else:
        theta = jnp.einsum("bij,bljr->blir", matrix, theta, precision="highest")
    theta = theta.reshape(left.shape[0], left.shape[1], 2, 2, right.shape[3])
    if reverse:
        theta = jnp.swapaxes(theta, 2, 3)
    bsz, left_dim, _, _, right_dim = theta.shape
    pair_matrix = theta.reshape(bsz, left_dim * 2, 2 * right_dim)
    return _split_pair_matrix_jax(
        pair_matrix,
        left_dim=int(left_dim),
        right_dim=int(right_dim),
        max_bond=max_bond,
        cutoff=cutoff,
    )


def _reconstruct_torch_mps_from_jax_rank_shards(
    rank_shards: Sequence[JAXMPSRankShardState],
    *,
    n_wires: int,
    max_bond: int | None,
    cutoff: float,
    complex_bytes: int,
    truncation_records: Sequence[Mapping[str, Any]],
) -> Any:
    import numpy as np

    torch = _require_torch()
    from ....simulation.mps import MPSConfig, MPSState, MPSTruncationRecord

    dtype = _torch_complex_dtype(complex_bytes)
    tensors_by_wire: dict[int, Any] = {}
    for shard in rank_shards:
        for wire, tensor in shard.local_tensors.items():
            tensors_by_wire[int(wire)] = torch.as_tensor(
                np.asarray(tensor).copy(), dtype=dtype, device=torch.device("cpu")
            )
    missing = [wire for wire in range(int(n_wires)) if wire not in tensors_by_wire]
    if missing:
        raise RuntimeError(
            f"Cannot reconstruct JAX sharded MPS facade; missing wires {missing}."
        )
    mps = MPSState(
        [tensors_by_wire[wire] for wire in range(int(n_wires))],
        config=MPSConfig(max_bond=max_bond, cutoff=float(cutoff)),
    )
    records = [
        MPSTruncationRecord(
            bond=int(record["bond"]),
            kept_rank=int(record["kept_rank"]),
            original_rank=int(record["original_rank"]),
            discarded_weight=float(record["discarded_weight"]),
            max_bond=record.get("max_bond"),
            cutoff=float(record.get("cutoff", 0.0)),
            source=str(record.get("source", "jax_sharded_mps_two_site")),
        )
        for record in truncation_records
    ]
    mps.truncation_records.extend(records)
    mps.truncation_errors.extend(
        float(record.discarded_weight)
        for record in records
        if float(record.discarded_weight) > 0
    )
    mps.orthogonality_center = int(n_wires) - 1 if n_wires else None
    return mps


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
