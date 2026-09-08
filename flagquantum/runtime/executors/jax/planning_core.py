"""Backend-neutral inputs and shared JAX distributed planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...distributed.backend_policy import DistributedBackendPolicy
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .runtime_environment import _require_jax


def _as_ir(program: Any) -> CircuitIR:
    return ensure_circuit_ir(program)


@dataclass(frozen=True)
class JAXDistributedQuantumPlan:
    """Machine-readable JAX distributed integration plan."""

    mode: str
    n_wires: int
    world_size: int
    local_world_size: int
    node_count: int
    backend_policy: DistributedBackendPolicy
    rank_ownership: tuple[Mapping[str, Any], ...]
    communication_tiers: Mapping[str, Any]
    local_memory_bytes_by_rank: tuple[int, ...]
    blockers: tuple[str, ...]
    gradient_blockers: tuple[str, ...]
    task_summary: Mapping[str, Any]
    intended_distribution_semantics: str = "sharded_across_ranks"
    distribution_semantics: str = "requires_runtime_summary"
    scalability_claim_allowed: bool = False
    interface: str = "torch"
    backend: str = "jax"

    def summary(self) -> dict[str, Any]:
        payload = {
            "executor": "jax_distributed_quantum_backend_plan",
            "backend": self.backend,
            "interface": self.interface,
            "claim_evidence_type": "plan_preflight",
            "jax_backend": self.backend_policy.jax_backend,
            "torch_backend": self.backend_policy.torch_backend,
            "distributed_backend_policy": self.backend_policy.summary(),
            "mode": self.mode,
            "n_wires": self.n_wires,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "distribution_semantics": self.distribution_semantics,
            "intended_distribution_semantics": self.intended_distribution_semantics,
            "scalability_claim_allowed": self.scalability_claim_allowed,
            "release_gate_allowed": False,
            "sharding_plan_available": self.intended_distribution_semantics
            == "sharded_across_ranks",
            "scalability_blockers": self.blockers,
            "gradient_blockers": self.gradient_blockers,
            "rank_ownership": self.rank_ownership,
            "local_memory_bytes_by_rank": self.local_memory_bytes_by_rank,
            "communication_tiers": dict(self.communication_tiers),
            "task_summary": dict(self.task_summary),
            "single_workload_sharding_required": True,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
            "status": "planned_not_executed",
        }
        return _attach_distributed_evidence_contract(payload)


def _jax_global_indices_by_rank_for_plan(plan: Any) -> Any:
    _, jnp = _require_jax()
    import numpy as np

    indices_by_rank: list[list[int]] = []
    sharded_wires = tuple(int(wire) for wire in plan.sharded_wires)
    if str(plan.distribution) == "qubit_address_sharded" and sharded_wires:
        for rank in range(int(plan.world_size)):
            coords = tuple(
                int(coord) for coord in plan.topology.rank_coordinates[int(rank)]
            )
            indices = []
            for basis in range(int(plan.total_amplitudes)):
                owned = True
                for coord, wire in zip(coords, sharded_wires):
                    bit = (basis >> (int(plan.n_wires) - int(wire) - 1)) & 1
                    if int(bit) != int(coord):
                        owned = False
                        break
                if owned:
                    indices.append(int(basis))
            indices_by_rank.append(indices)
    else:
        for shard in plan.shards:
            indices_by_rank.append(
                list(range(int(shard.amplitude_start), int(shard.amplitude_end)))
            )
    lengths = {len(indices) for indices in indices_by_rank}
    if len(lengths) != 1:
        raise RuntimeError(
            "JAX pmap statevector backward requires equal local shard sizes."
        )
    dtype = jnp.int64 if int(plan.total_amplitudes) > 2**31 else jnp.int32
    return jnp.asarray(np.asarray(indices_by_rank), dtype=dtype)
