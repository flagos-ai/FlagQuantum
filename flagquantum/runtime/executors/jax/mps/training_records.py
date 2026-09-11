"""MPS training, parameter-flow, gate-assignment, and rank-shard records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from ..runtime_environment import _jax_array_nbytes
from .evidence import _build_mps_backward_resource_evidence


@dataclass(frozen=True)
class JAXShardedMPSTrainingPlan:
    """Preflight plan for site-sharded MPS parameter gradients."""

    mode: str
    backend: str
    backward_execution: str
    distribution_semantics: str
    intended_distribution_semantics: str
    scalability_claim_allowed: bool
    gradient_ready: bool
    local_development_gradient_ready: bool
    world_size: int
    local_world_size: int
    node_count: int
    n_wires: int
    batch_size: int
    max_bond: int | None
    cutoff: float
    boundary_sync_count: int
    static_blockers: tuple[str, ...]
    device_blockers: tuple[str, ...]
    blockers: tuple[str, ...]
    jax_plan_summary: Mapping[str, Any]
    parameter_flow_summary: Mapping[str, Any]
    device_summary: Mapping[str, Any]
    inspected_devices: bool

    def summary(self) -> dict[str, Any]:
        jax_summary = dict(self.jax_plan_summary)
        communication_tiers = dict(jax_summary.get("communication_tiers", {}))
        parameter_flow = dict(self.parameter_flow_summary)
        rank_memory = tuple(jax_summary.get("local_memory_bytes_by_rank", ()))
        site_ownership = tuple(jax_summary.get("rank_ownership", ()))
        bond_ownership = tuple(communication_tiers.get("boundary_edges", ()))
        parameter_ownership = tuple(
            {
                "rank": rank,
                "owned_parameter_count": int(count),
                "ownership_semantics": "rank_owned_parameter_slice",
            }
            for rank, count in enumerate(
                parameter_flow.get("rank_parameter_counts", ())
            )
            if int(count) > 0
        )
        boundary_routes = bond_ownership if bond_ownership else "not_required"
        rank_summaries = tuple(
            {
                "rank": rank,
                "wires": tuple(
                    int(wire)
                    for wire in (
                        site_ownership[rank].get("wires", ())
                        if rank < len(site_ownership)
                        else ()
                    )
                ),
                "local_tensor_bytes": int(rank_memory[rank]),
                "tensor_bytes_by_wire": {},
            }
            for rank in range(self.world_size)
        )
        planned_protocols = tuple(
            {
                "left_rank": int(edge["left_rank"]),
                "right_rank": int(edge["right_rank"]),
                "left_wire": int(edge["left_wire"]),
                "right_wire": int(edge["right_wire"]),
                "tier": str(edge.get("tier", "unknown")),
                "estimated_transfer_bytes": int(
                    edge.get("estimated_transfer_bytes", 0)
                ),
            }
            for edge in bond_ownership
        )
        parameter_gradient_bytes = tuple(
            int(count) * 8 for count in parameter_flow.get("rank_parameter_counts", ())
        )
        planned_truncation = (
            tuple({"bond": int(edge["left_wire"])} for edge in bond_ownership)
            if self.cutoff > 0.0
            else ()
        )
        resource_evidence = _build_mps_backward_resource_evidence(
            rank_summaries,
            planned_protocols,
            (),
            parameter_gradient_bytes,
            planned_truncation,
            world_size=self.world_size,
            local_world_size=self.local_world_size,
            node_count=self.node_count,
            execution_scope="plan_preflight",
            communication_executed=False,
        )
        payload = {
            "planner": "jax_sharded_mps_training",
            "claim_evidence_type": "plan_preflight",
            "mode": self.mode,
            "state_mode": "jax_sharded_mps",
            "backend": self.backend,
            "backward_execution": self.backward_execution,
            "distribution_semantics": self.distribution_semantics,
            "intended_distribution_semantics": self.intended_distribution_semantics,
            "scalability_claim_allowed": self.scalability_claim_allowed,
            "release_gate_allowed": False,
            "sharding_plan_available": self.intended_distribution_semantics
            == "sharded_across_ranks",
            "scalability_blockers": self.blockers,
            "gradient_ready": self.gradient_ready,
            "parameter_gradient_ready": self.gradient_ready,
            "gradient_distribution_semantics": (
                "sharded_across_ranks" if self.gradient_ready else "incomplete"
            ),
            "mps_forward_distribution_semantics": self.intended_distribution_semantics,
            "mps_backward_distribution_semantics": (
                "sharded_across_ranks" if self.gradient_ready else "incomplete"
            ),
            "site_shard_ownership": site_ownership,
            "bond_shard_ownership": bond_ownership,
            "parameter_gradient_ownership": parameter_ownership,
            "boundary_gradient_ownership": (
                tuple(
                    {
                        "left_rank": int(edge["left_rank"]),
                        "right_rank": int(edge["right_rank"]),
                        "ownership_semantics": "adjacent_rank_boundary_gradient",
                    }
                    for edge in bond_ownership
                )
                if bond_ownership
                else "not_required"
            ),
            "boundary_adjoint_exchange": {
                "status": "planned_not_executed",
                "route": "adjacent_rank_adjoint_boundary_exchange",
            },
            "boundary_adjoint_exchange_executor": {
                "local_cpu_probe_available": True,
                "production_transport_available": False,
                "claim_evidence_type": "development_smoke",
                "scalability_claim_allowed": False,
            },
            "boundary_gradient_routes": boundary_routes,
            "canonicalization_backward_strategy": "pending",
            "truncation_gradient_metadata": (
                {"status": "not_required", "cutoff": self.cutoff}
                if self.cutoff <= 0.0
                else {"status": "pending", "cutoff": self.cutoff}
            ),
            "mps_backward_resource_evidence": resource_evidence,
            "mps_backward_memory_plan": resource_evidence["mps_backward_memory_plan"],
            "mps_backward_communication_plan": {
                **resource_evidence["mps_backward_communication_plan"],
                "boundary_adjoint_exchange": "planned_not_executed",
                "boundary_gradient_routes": boundary_routes,
            },
            "optimizer_update_semantics": "not_measured",
            "optimizer_update_ownership": (),
            "fallback_semantics": (
                "local_simulation"
                if self.backend == "local_simulated"
                else "none_planned"
            ),
            "local_development_gradient_ready": self.local_development_gradient_ready,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.batch_size,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "rank_ownership": site_ownership,
            "local_memory_bytes_by_rank": rank_memory,
            "communication_tiers": communication_tiers,
            "estimated_transfer_bytes": int(
                communication_tiers.get("estimated_transfer_bytes", 0)
            ),
            "boundary_sync_count": self.boundary_sync_count,
            "static_blockers": self.static_blockers,
            "device_blockers": self.device_blockers,
            "blockers": self.blockers,
            "parameter_flow": parameter_flow,
            "parameter_flow_blockers": tuple(parameter_flow.get("blockers", ())),
            "jax_distributed_plan": jax_summary,
            "device_summary": dict(self.device_summary),
            "inspected_devices": self.inspected_devices,
            "single_workload_sharding_required": True,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
            "full_mps_reconstruction_allowed_for_claimable_backward": False,
            "silent_statevector_fallback_allowed": False,
        }
        return _attach_mps_backward_readiness(payload)


@dataclass(frozen=True)
class JAXShardedMPSParameterGateAssignment:
    """Ownership and gradient route for one parameterized MPS instruction."""

    instruction_index: int
    name: str
    wires: tuple[int, ...]
    parameter_names: tuple[str, ...]
    owner_rank: int | None
    touched_ranks: tuple[int, ...]
    locality: str
    gradient_route: str
    communication_tier: str
    blockers: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "instruction_index": self.instruction_index,
            "name": self.name,
            "wires": self.wires,
            "parameter_names": self.parameter_names,
            "owner_rank": self.owner_rank,
            "touched_ranks": self.touched_ranks,
            "locality": self.locality,
            "gradient_route": self.gradient_route,
            "communication_tier": self.communication_tier,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class JAXShardedMPSParameterFlowPlan:
    """Static parameter-gate flow plan for sharded MPS backward."""

    world_size: int
    local_world_size: int
    node_count: int
    n_wires: int
    site_shard_ownership: tuple[Mapping[str, Any], ...]
    bond_shard_ownership: tuple[Mapping[str, Any], ...]
    assignments: tuple[JAXShardedMPSParameterGateAssignment, ...]
    rank_parameter_counts: tuple[int, ...]
    boundary_parameter_edges: tuple[Mapping[str, Any], ...]
    blockers: tuple[str, ...]

    @property
    def parameter_gate_count(self) -> int:
        return len(self.assignments)

    @property
    def rank_local_parameter_gate_count(self) -> int:
        return sum(1 for item in self.assignments if item.locality == "site_local")

    @property
    def boundary_parameter_gate_count(self) -> int:
        return sum(1 for item in self.assignments if item.locality == "boundary")

    @property
    def unsupported_parameter_gate_count(self) -> int:
        return sum(1 for item in self.assignments if item.locality == "unsupported")

    def summary(self) -> dict[str, Any]:
        rank_local_ready = bool(
            self.world_size > 1
            and self.parameter_gate_count > 0
            and self.boundary_parameter_gate_count == 0
            and self.unsupported_parameter_gate_count == 0
        )
        parameter_ownership = tuple(
            {
                "rank": rank,
                "owned_parameter_count": int(count),
                "ownership_semantics": "rank_owned_parameter_slice",
            }
            for rank, count in enumerate(self.rank_parameter_counts)
            if int(count) > 0
        )
        payload = {
            "planner": "jax_sharded_mps_parameter_flow",
            "claim_evidence_type": "plan_preflight",
            "state_mode": "jax_sharded_mps",
            "distribution_semantics": (
                "requires_runtime_summary"
                if self.world_size > 1
                else "replicated_single_rank"
            ),
            "intended_distribution_semantics": "sharded_across_ranks",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": self.world_size > 1,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "parameter_gate_count": self.parameter_gate_count,
            "rank_local_parameter_gate_count": self.rank_local_parameter_gate_count,
            "boundary_parameter_gate_count": self.boundary_parameter_gate_count,
            "unsupported_parameter_gate_count": self.unsupported_parameter_gate_count,
            "rank_local_parameter_vjp_ready": rank_local_ready,
            "boundary_parameter_vjp_ready": False,
            "rank_parameter_counts": self.rank_parameter_counts,
            "boundary_parameter_edges": self.boundary_parameter_edges,
            "assignments": tuple(item.summary() for item in self.assignments),
            "gradient_reduction_plan": {
                "parameter_gradient_layout": "global_parameter_tensor_with_rank_owned_nonzero_slices",
                "rank_local_vjp": "owner_rank_computes_site_local_parameter_pullback",
                "boundary_vjp": (
                    "adjacent_rank_adjoint_boundary_exchange_required"
                    if self.boundary_parameter_gate_count
                    else "not_required"
                ),
                "zero_fill_non_owned_gradients": True,
                "optimizer_update": "all_reduce_or_owner_writeback_required",
            },
            "mps_forward_distribution_semantics": (
                "sharded_across_ranks"
                if self.world_size > 1
                else "replicated_single_rank"
            ),
            "mps_backward_distribution_semantics": "planned_sharded_across_ranks",
            "site_shard_ownership": self.site_shard_ownership,
            "bond_shard_ownership": self.bond_shard_ownership,
            "parameter_gradient_ownership": parameter_ownership,
            "boundary_gradient_ownership": (
                "planned_from_bond_shard_ownership"
                if self.bond_shard_ownership
                else "not_required"
            ),
            "boundary_adjoint_exchange": {
                "status": "planned_not_executed",
                "route": "adjacent_rank_adjoint_boundary_exchange",
            },
            "boundary_gradient_routes": (
                self.bond_shard_ownership
                if self.bond_shard_ownership
                else "not_required"
            ),
            "canonicalization_backward_strategy": "pending",
            "truncation_gradient_metadata": "not_measured",
            "mps_backward_memory_plan": {},
            "mps_backward_communication_plan": {
                "boundary_gradient_routes": (
                    self.bond_shard_ownership
                    if self.bond_shard_ownership
                    else "not_required"
                ),
            },
            "optimizer_update_semantics": "not_measured",
            "optimizer_update_ownership": (),
            "fallback_semantics": "none_planned",
            "blockers": self.blockers,
            "single_workload_sharding_required": True,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_mps_backward_readiness(payload)


@dataclass
class JAXMPSRankShardState:
    """Rank-local JAX MPS site tensors owned by one shard."""

    rank: int
    wires: tuple[int, ...]
    local_tensors: Mapping[int, Any]

    def summary(self) -> dict[str, Any]:
        tensor_bytes = sum(
            _jax_array_nbytes(tensor) for tensor in self.local_tensors.values()
        )
        return {
            "rank": self.rank,
            "wires": self.wires,
            "local_tensor_wires": tuple(
                sorted(int(wire) for wire in self.local_tensors)
            ),
            "local_tensor_count": len(self.local_tensors),
            "local_tensor_bytes": int(tensor_bytes),
            "tensor_bytes_by_wire": {
                int(wire): _jax_array_nbytes(tensor)
                for wire, tensor in sorted(self.local_tensors.items())
            },
            "tensor_shapes": {
                int(wire): tuple(int(dim) for dim in tensor.shape)
                for wire, tensor in sorted(self.local_tensors.items())
            },
            "dtype": (
                str(next(iter(self.local_tensors.values())).dtype)
                if self.local_tensors
                else "empty"
            ),
        }
