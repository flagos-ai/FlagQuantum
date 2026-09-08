"""Local MPS boundary-adjoint exchange execution and measurement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _summarize_local_mps_boundary_adjoint_exchange(
    records: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    expected_boundary_edges: int,
    execution_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate local CPU boundary-adjoint exchange execution evidence."""

    normalized_records = tuple(dict(record) for record in records)
    blockers = [str(item) for item in execution_blockers if str(item)]
    world_size = int(world_size)
    expected_boundary_edges = int(expected_boundary_edges)
    if world_size <= 1:
        blockers.append("mps_boundary_adjoint_exchange_world_size_must_exceed_one")
    if expected_boundary_edges < 0:
        blockers.append("mps_boundary_adjoint_exchange_expected_edge_count_invalid")
    expected_records = max(0, expected_boundary_edges) * 2
    if expected_records and not normalized_records:
        blockers.append("mps_boundary_adjoint_exchange_routes_missing")
    if len(normalized_records) != expected_records:
        blockers.append("mps_boundary_adjoint_exchange_route_count_mismatch")

    local_memory_by_rank = [0 for _ in range(max(1, world_size))]
    communication_bytes = 0
    seen_routes: set[tuple[str, str]] = set()
    for index, record in enumerate(normalized_records):
        record_id = str(record.get("boundary_edge_id", f"record_{index}"))
        direction = str(record.get("direction", ""))
        route_key = (record_id, direction)
        if (
            not record.get("boundary_edge_id")
            or record.get("bond_id") is None
            or direction not in {"left_to_right", "right_to_left"}
            or route_key in seen_routes
        ):
            blockers.append(f"mps_boundary_adjoint_exchange_route_missing:{record_id}")
        seen_routes.add(route_key)
        try:
            source_rank = int(record.get("source_rank", -1))
            target_rank = int(record.get("target_rank", -1))
        except (TypeError, ValueError):
            source_rank = -1
            target_rank = -1
        owned_ranges = record.get("owned_site_ranges")
        valid_ranks = (
            0 <= source_rank < world_size
            and 0 <= target_rank < world_size
            and source_rank != target_rank
        )
        valid_ranges = bool(
            isinstance(owned_ranges, Mapping)
            and owned_ranges.get("source")
            and owned_ranges.get("target")
        )
        if not valid_ranks or not valid_ranges:
            blockers.append(
                f"mps_boundary_adjoint_exchange_rank_ownership_missing:{record_id}"
            )
        shape = record.get("boundary_tensor_shape")
        try:
            valid_shape = bool(
                isinstance(shape, (tuple, list))
                and shape
                and all(int(dim) > 0 for dim in shape)
            )
        except (TypeError, ValueError):
            valid_shape = False
        if not valid_shape:
            blockers.append(
                f"mps_boundary_adjoint_exchange_tensor_shape_missing:{record_id}"
            )
        if not str(record.get("boundary_tensor_dtype", "")):
            blockers.append(
                f"mps_boundary_adjoint_exchange_tensor_dtype_missing:{record_id}"
            )
        if (
            str(record.get("communication_primitive", ""))
            != "local_cpu_point_to_point_copy"
        ):
            blockers.append(
                f"mps_boundary_adjoint_exchange_primitive_missing:{record_id}"
            )
        try:
            record_communication_bytes = int(record.get("communication_bytes", 0))
            record_memory_bytes = int(record.get("local_memory_bytes", 0))
        except (TypeError, ValueError):
            record_communication_bytes = 0
            record_memory_bytes = 0
        if record_communication_bytes <= 0:
            blockers.append(
                f"mps_boundary_adjoint_exchange_communication_bytes_missing:{record_id}"
            )
        if record_memory_bytes <= 0:
            blockers.append(
                f"mps_boundary_adjoint_exchange_local_memory_missing:{record_id}"
            )
        ownership = record.get("boundary_gradient_ownership")
        if not (
            isinstance(ownership, Mapping)
            and ownership.get("ownership_semantics")
            and ownership.get("owner_rank") == target_rank
        ):
            blockers.append(f"mps_boundary_gradient_ownership_missing:{record_id}")
        if record.get("boundary_adjoint_exchange_status") != "executed":
            blockers.append(f"mps_boundary_adjoint_exchange_not_executed:{record_id}")
        record_blockers = tuple(
            str(item) for item in record.get("blockers", ()) or () if str(item)
        )
        if record_blockers:
            blockers.extend(record_blockers)
        communication_bytes += max(0, record_communication_bytes)
        if 0 <= target_rank < len(local_memory_by_rank):
            local_memory_by_rank[target_rank] += max(0, record_memory_bytes)

    execution_blockers_out = tuple(dict.fromkeys(blockers))
    if execution_blockers_out:
        status = "blocked"
    elif expected_boundary_edges == 0:
        status = "not_required"
    else:
        status = "local_cpu_executed"
    claim_blockers = (
        (
            "mps_boundary_adjoint_exchange_local_cpu_only",
            "mps_boundary_adjoint_values_are_probe_only",
        )
        if expected_boundary_edges > 0
        else ()
    ) + ("mps_full_sharded_backward_executor_pending",)
    all_blockers = tuple(dict.fromkeys((*execution_blockers_out, *claim_blockers)))
    return {
        "status": status,
        "valid": not execution_blockers_out,
        "execution_scope": "development_cpu",
        "evidence_scope": "boundary_exchange_probe_only",
        "claim_evidence_type": "development_smoke",
        "distribution_semantics": (
            "sharded_across_ranks" if world_size > 1 else "replicated_single_rank"
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "world_size": world_size,
        "expected_boundary_edges": expected_boundary_edges,
        "directed_exchange_count": len(normalized_records),
        "communication_bytes": int(communication_bytes),
        "local_memory_bytes_by_rank": tuple(local_memory_by_rank),
        "communication_primitive": "local_cpu_point_to_point_copy",
        "boundary_gradient_ownership": tuple(
            record.get("boundary_gradient_ownership")
            for record in normalized_records
            if record.get("boundary_gradient_ownership")
        ),
        "boundary_gradient_routes": tuple(
            {
                "boundary_edge_id": record.get("boundary_edge_id"),
                "source_rank": record.get("source_rank"),
                "target_rank": record.get("target_rank"),
                "communication_primitive": record.get("communication_primitive"),
                "communication_bytes": record.get("communication_bytes"),
            }
            for record in normalized_records
        ),
        "records": normalized_records,
        "execution_blockers": execution_blockers_out,
        "blockers": all_blockers,
    }


def _execute_local_mps_boundary_adjoint_exchange(
    rank_tensors: Mapping[int, Mapping[int, Any]],
    boundary_protocols: Sequence[Mapping[str, Any]],
    shard_plans: Sequence[Any],
) -> dict[str, Any]:
    """Execute a host-CPU probe of directed boundary-adjoint transfers."""

    import numpy as np

    ownership = {
        int(shard.rank): tuple(int(wire) for wire in shard.wires)
        for shard in shard_plans
    }
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for edge_index, protocol in enumerate(boundary_protocols):
        try:
            left_rank = int(protocol["left_rank"])
            right_rank = int(protocol["right_rank"])
            left_wire = int(protocol["left_wire"])
            right_wire = int(protocol["right_wire"])
        except (KeyError, TypeError, ValueError):
            blockers.append(
                f"mps_boundary_adjoint_exchange_route_missing:edge_{edge_index}"
            )
            continue
        edge_id = f"edge_{edge_index}:{left_wire}-{right_wire}"
        directions = (
            (left_rank, right_rank, left_wire, right_wire, "left_to_right"),
            (right_rank, left_rank, right_wire, left_wire, "right_to_left"),
        )
        for source_rank, target_rank, source_wire, target_wire, direction in directions:
            source_tensor = rank_tensors.get(source_rank, {}).get(source_wire)
            if source_tensor is None:
                blockers.append(
                    f"mps_boundary_adjoint_exchange_tensor_missing:{edge_id}:{direction}"
                )
                continue
            shape = tuple(int(dim) for dim in getattr(source_tensor, "shape", ()))
            dtype_name = str(getattr(source_tensor, "dtype", ""))
            if not shape:
                blockers.append(
                    f"mps_boundary_adjoint_exchange_tensor_shape_missing:{edge_id}:{direction}"
                )
                continue
            try:
                source_adjoint = np.ones(shape, dtype=np.dtype(dtype_name))
                received_adjoint = np.array(source_adjoint, copy=True)
            except (TypeError, ValueError):
                blockers.append(
                    f"mps_boundary_adjoint_exchange_tensor_dtype_missing:{edge_id}:{direction}"
                )
                continue
            if not np.array_equal(source_adjoint, received_adjoint):
                blockers.append(
                    f"mps_boundary_adjoint_exchange_copy_validation_failed:{edge_id}:{direction}"
                )
                continue
            transfer_bytes = int(received_adjoint.nbytes)
            records.append(
                {
                    "boundary_edge_id": edge_id,
                    "direction": direction,
                    "source_rank": source_rank,
                    "target_rank": target_rank,
                    "source_wire": source_wire,
                    "target_wire": target_wire,
                    "owned_site_ranges": {
                        "source": ownership.get(source_rank, ()),
                        "target": ownership.get(target_rank, ()),
                    },
                    "bond_id": left_wire,
                    "boundary_id": f"bond_{left_wire}_{right_wire}",
                    "boundary_tensor_shape": shape,
                    "boundary_tensor_dtype": dtype_name,
                    "communication_primitive": "local_cpu_point_to_point_copy",
                    "local_memory_bytes": transfer_bytes,
                    "communication_bytes": transfer_bytes,
                    "boundary_gradient_ownership": {
                        "source_rank": source_rank,
                        "target_rank": target_rank,
                        "owner_rank": target_rank,
                        "ownership_semantics": "received_boundary_adjoint",
                    },
                    "boundary_adjoint_exchange_status": "executed",
                    "adjoint_source": "synthetic_unit_boundary_probe",
                    "blockers": (),
                }
            )
    return _summarize_local_mps_boundary_adjoint_exchange(
        records,
        world_size=len(shard_plans),
        expected_boundary_edges=len(boundary_protocols),
        execution_blockers=blockers,
    )
