"""Sharded MPS forward execution result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....distributed.backend_policy import DistributedBackendPolicy
from ..mps_kernels import _reconstruct_torch_mps_from_jax_rank_shards
from ..planning_core import JAXDistributedQuantumPlan
from ..release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from ..runtime_environment import _require_torch
from .training_records import JAXMPSRankShardState


@dataclass
class JAXShardedMPSResult:
    """Executable JAX MPS result with site tensors sharded across ranks."""

    rank_shards: tuple[JAXMPSRankShardState, ...]
    shard_plans: tuple[Any, ...]
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    n_wires: int
    bsz: int
    complex_bytes: int
    max_bond: int | None
    cutoff: float
    owned_instruction_count: int
    sharded_kernel_count: int
    boundary_sync_count: int
    boundary_transfer_bytes: int
    unsupported_instruction_count: int
    truncation_error: float
    max_truncation_error: float
    truncation_records: tuple[Mapping[str, Any], ...]
    boundary_protocols: tuple[Mapping[str, Any], ...]
    full_mps_reconstruction_count: int = 0
    statevector_facade_count: int = 0
    _torch_mps_cache: Any | None = None
    _torch_state_cache: Any | None = None

    def to_mps(self, *, refresh: bool = False) -> Any:
        """Gather rank-owned MPS tensors into a local torch MPS facade."""

        if self._torch_mps_cache is not None and not refresh:
            return self._torch_mps_cache
        self._torch_mps_cache = _reconstruct_torch_mps_from_jax_rank_shards(
            self.rank_shards,
            n_wires=self.n_wires,
            max_bond=self.max_bond,
            cutoff=self.cutoff,
            complex_bytes=self.complex_bytes,
            truncation_records=self.truncation_records,
        )
        self.full_mps_reconstruction_count += 1
        return self._torch_mps_cache

    def to_statevector(self, *, refresh: bool = False) -> Any:
        if self._torch_state_cache is not None and not refresh:
            return self._torch_state_cache
        self._torch_state_cache = self.to_mps(refresh=refresh).to_statevector()
        self.statevector_facade_count += 1
        return self._torch_state_cache

    state = to_statevector
    wavefunction = to_statevector

    def probabilities(self) -> Any:
        torch = _require_torch()
        return torch.abs(self.to_statevector()) ** 2

    probability = probabilities

    def expectation_z(self, wire: int | None = None) -> Any:
        return self.to_mps().expectation_z(wire)

    def summary(self) -> dict[str, Any]:
        is_sharded = len(self.rank_shards) > 1
        local_memory = tuple(
            int(shard.summary()["local_tensor_bytes"]) for shard in self.rank_shards
        )
        jax_plan_summary = self.jax_plan.summary()
        bond_ownership = tuple(
            jax_plan_summary.get("communication_tiers", {}).get("boundary_edges", ())
        )
        intra = sum(
            int(item["estimated_transfer_bytes"])
            for item in self.boundary_protocols
            if item.get("tier") == "intra_node"
        )
        inter = sum(
            int(item["estimated_transfer_bytes"])
            for item in self.boundary_protocols
            if item.get("tier") == "inter_node"
        )
        blockers = []
        if not is_sharded:
            blockers.append("world_size_is_one")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        blockers.extend(
            [
                "production_jax_pmap_mps_site_executor_pending",
                "production_jax_mps_boundary_transport_pending",
            ]
        )
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        payload = {
            "executor": "jax_sharded_mps_executor",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_mps",
            "backend": "jax",
            "interface": "torch",
            "jax_backend": self.backend_policy.jax_backend,
            "torch_backend": self.backend_policy.torch_backend,
            "distributed_backend_policy": self.backend_policy.summary(),
            "distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "forward_only_jax_sharded_mps",
            "gradient_blockers": (
                "mps_boundary_adjoint_exchange_pending",
                "mps_parameter_gradient_ownership_pending",
                "mps_optimizer_update_ownership_pending",
            ),
            "mps_forward_distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "mps_backward_distribution_semantics": "incomplete",
            "site_shard_ownership": tuple(
                shard.summary() for shard in self.rank_shards
            ),
            "bond_shard_ownership": bond_ownership,
            "parameter_gradient_ownership": (),
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
            "boundary_gradient_routes": (
                bond_ownership if bond_ownership else "not_required"
            ),
            "canonicalization_backward_strategy": "pending",
            "truncation_gradient_metadata": (
                {"status": "pending", "records": self.truncation_records}
                if self.truncation_records
                else {"status": "not_required", "cutoff": self.cutoff}
            ),
            "mps_backward_memory_plan": {
                "local_memory_bytes_by_rank": local_memory,
                "communication_buffer_bytes": int(self.boundary_transfer_bytes),
            },
            "mps_backward_communication_plan": {
                "estimated_transfer_bytes": int(self.boundary_transfer_bytes),
                "boundary_adjoint_exchange": "planned_not_executed",
                "boundary_gradient_routes": (
                    bond_ownership if bond_ownership else "not_required"
                ),
            },
            "optimizer_update_semantics": "not_measured",
            "optimizer_update_ownership": (),
            "fallback_semantics": (
                (
                    "full_local_mps_state_view",
                    "statevector_fallback_for_correctness_inspection",
                )
                if self.statevector_facade_count > 0
                else (
                    "full_local_mps_state_view"
                    if self.full_mps_reconstruction_count > 0
                    else (
                        "local_simulation"
                        if self.backend_policy.profile != "production"
                        else "none"
                    )
                )
            ),
            "world_size": len(self.rank_shards),
            "local_world_size": self.jax_plan.local_world_size,
            "node_count": self.jax_plan.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.bsz,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "owned_instruction_count": self.owned_instruction_count,
            "sharded_kernel_count": self.sharded_kernel_count,
            "boundary_sync_count": self.boundary_sync_count,
            "unsupported_instruction_count": self.unsupported_instruction_count,
            "boundary_transfer_bytes": int(self.boundary_transfer_bytes),
            "communication_tiers": {
                "model": "jax_mps_site_sharded_boundary_exchange",
                "boundary_edge_count": len(self.boundary_protocols),
                "estimated_transfer_bytes": int(self.boundary_transfer_bytes),
                "intra_node_communication_bytes": int(intra),
                "inter_node_communication_bytes": int(inter),
                "boundary_protocols": self.boundary_protocols,
            },
            "intra_node_communication_bytes": int(intra),
            "inter_node_communication_bytes": int(inter),
            "local_memory_bytes_by_rank": local_memory,
            "rank_shards": tuple(shard.summary() for shard in self.rank_shards),
            "truncation_steps": len(self.truncation_records),
            "truncation_error": float(self.truncation_error),
            "max_truncation_error": float(self.max_truncation_error),
            "truncation_records": self.truncation_records,
            "full_mps_reconstruction_count": self.full_mps_reconstruction_count,
            "statevector_facade_count": self.statevector_facade_count,
            "silent_statevector_fallback": False,
            "full_state_fallback_count": 0,
            "jax_distributed_plan": jax_plan_summary,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_mps_backward_readiness(payload)
