# ruff: noqa: F401, F821
"""Statevector planning, shard-state, and execution result records."""

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


def _statevector_memory_plan_for_claimability(
    *,
    per_rank_shard_bytes: Sequence[int],
    peak_buffer_bytes: int,
    communication_buffer_count: int,
) -> dict[str, Any]:
    return {
        "state_partition": "amplitude_or_qubit_address_shards",
        "per_rank_shard_bytes": tuple(int(item) for item in per_rank_shard_bytes),
        "communication_buffer_bytes": int(peak_buffer_bytes),
        "communication_buffer_count": int(communication_buffer_count),
        "requires_per_rank_memory_evidence": True,
    }


def _statevector_communication_plan_for_claimability(
    *,
    topology_summary: Mapping[str, Any],
    communication_execution: str,
    blockers: Sequence[str],
) -> dict[str, Any]:
    communications = tuple(
        str(item) for item in topology_summary.get("communications", ()) or ()
    )
    return {
        "communication_execution": str(communication_execution),
        "transport_patterns": communications,
        "collective_blockers": tuple(
            str(item)
            for item in blockers
            if "transport" in str(item)
            or "collective" in str(item)
            or "all_to_all" in str(item)
        ),
        "topology_dependency": "collective_route_depends_on_jax_runtime",
        "world_size": int(topology_summary.get("world_size", 1) or 1),
        "local_world_size": int(topology_summary.get("local_world_size", 1) or 1),
        "node_count": int(topology_summary.get("node_count", 1) or 1),
        "intra_node_communication_bytes": int(
            topology_summary.get("intra_node_communication_bytes", 0) or 0
        ),
        "inter_node_communication_bytes": int(
            topology_summary.get("inter_node_communication_bytes", 0) or 0
        ),
    }


@dataclass(frozen=True)
class JAXShardedStatevectorTrainingPlan:
    """Preflight plan for amplitude-sharded statevector parameter gradients."""

    mode: str
    backend: str
    backward_execution: str
    distribution_semantics: str
    scalability_claim_allowed: bool
    gradient_ready: bool
    world_size: int
    local_world_size: int
    node_count: int
    n_wires: int
    batch_size: int
    static_blockers: tuple[str, ...]
    device_blockers: tuple[str, ...]
    blockers: tuple[str, ...]
    statevector_plan_summary: Mapping[str, Any]
    jax_plan_summary: Mapping[str, Any]
    device_summary: Mapping[str, Any]
    inspected_devices: bool

    def summary(self) -> dict[str, Any]:
        statevector_summary = dict(self.statevector_plan_summary)
        rank_memory = tuple(
            int(item.get("local_state_bytes", 0))
            for item in statevector_summary.get("rank_shards", ())
            if isinstance(item, Mapping)
        )
        topology_summary = dict(statevector_summary.get("topology", {}))
        if not topology_summary:
            topology_summary = {
                "world_size": self.world_size,
                "local_world_size": self.local_world_size,
                "node_count": self.node_count,
                "communications": (),
                "intra_node_communication_bytes": int(
                    statevector_summary.get("intra_node_communication_bytes", 0)
                ),
                "inter_node_communication_bytes": int(
                    statevector_summary.get("inter_node_communication_bytes", 0)
                ),
            }
        memory_plan = _statevector_memory_plan_for_claimability(
            per_rank_shard_bytes=rank_memory,
            peak_buffer_bytes=int(statevector_summary.get("peak_buffer_bytes", 0)),
            communication_buffer_count=int(
                statevector_summary.get("communication_buffer_count", 0)
            ),
        )
        communication_plan = _statevector_communication_plan_for_claimability(
            topology_summary=topology_summary,
            communication_execution=self.backward_execution,
            blockers=self.blockers,
        )
        payload = {
            "planner": "jax_sharded_statevector_training",
            "claim_evidence_type": "plan_preflight",
            "mode": self.mode,
            "state_mode": "jax_sharded_statevector",
            "backend": self.backend,
            "backward_execution": self.backward_execution,
            "distribution_semantics": self.distribution_semantics,
            "forward_distribution_semantics": self.distribution_semantics,
            "backward_distribution_semantics": (
                "sharded_across_ranks" if self.gradient_ready else "incomplete"
            ),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": self.distribution_semantics
            == "sharded_across_ranks",
            "production_training_preflight_ready": self.scalability_claim_allowed,
            "scalability_blockers": self.blockers,
            "gradient_ready": self.gradient_ready,
            "parameter_gradient_ready": self.gradient_ready,
            "optimizer_update_semantics": "not_measured",
            "backward_uses_full_state_replay": False,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.batch_size,
            "rank_shards": tuple(statevector_summary.get("rank_shards", ())),
            "local_memory_bytes_by_rank": rank_memory,
            "memory_plan": memory_plan,
            "estimated_transfer_bytes": int(
                statevector_summary.get("estimated_transfer_bytes", 0)
            ),
            "intra_node_communication_bytes": int(
                statevector_summary.get("intra_node_communication_bytes", 0)
            ),
            "inter_node_communication_bytes": int(
                statevector_summary.get("inter_node_communication_bytes", 0)
            ),
            "communication_plan": communication_plan,
            "communication_tiers": {
                "model": "jax_statevector_backward_collectives",
                "communication_execution": self.backward_execution,
                "topology_dependency": "collective_route_depends_on_jax_runtime",
                "communications": tuple(topology_summary.get("communications", ())),
            },
            "static_blockers": self.static_blockers,
            "device_blockers": self.device_blockers,
            "blockers": self.blockers,
            "statevector_plan": statevector_summary,
            "jax_distributed_plan": dict(self.jax_plan_summary),
            "device_summary": dict(self.device_summary),
            "inspected_devices": self.inspected_devices,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_statevector_claimability(payload)


@dataclass
class JAXStatevectorShardState:
    """Rank-local JAX array shard for amplitude-sharded statevector execution."""

    rank: int
    shard: Any
    amplitudes: Any
    global_indices: Any
    global_indices_tuple: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "amplitude_start": self.shard.amplitude_start,
            "amplitude_end": self.shard.amplitude_end,
            "local_amplitudes": self.shard.local_amplitudes,
            "global_index_count": len(self.global_indices_tuple),
            "shape": tuple(int(item) for item in self.amplitudes.shape),
            "device": _jax_array_device_name(self.amplitudes),
            "dtype": str(self.amplitudes.dtype),
        }


@dataclass
class JAXShardedStatevectorResult:
    """Executable JAX statevector result with one logical state sharded by rank."""

    shards: tuple[JAXStatevectorShardState, ...]
    plan: Any
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    local_gate_count: int
    distributed_gate_count: int
    simulated_communication_count: int
    simulated_communication_bytes: int
    full_state_reconstruction_count: int = 0
    _torch_state_cache: Any | None = None

    def state(self, *, refresh: bool = False) -> Any:
        """Return a torch statevector facade for correctness checks and local APIs.

        Runtime execution stays shard-local. This method is intentionally a
        facade and increments ``full_state_reconstruction_count`` so benchmarks
        cannot mistake final inspection for a production all-gather-free path.
        """

        if self._torch_state_cache is not None and not refresh:
            return self._torch_state_cache
        self._torch_state_cache = _reconstruct_torch_state_from_jax_shards(
            self.shards, self.plan
        )
        self.full_state_reconstruction_count += 1
        return self._torch_state_cache

    wavefunction = state

    def probabilities(self) -> Any:
        torch = _require_torch()
        state = self.state()
        return torch.abs(state) ** 2

    probability = probabilities

    def expectation_z(self, wire: int | None = None) -> Any:
        torch = _require_torch()
        state = self.state()
        probabilities = torch.abs(state) ** 2
        wires = tuple(range(self.plan.n_wires)) if wire is None else (int(wire),)
        values = []
        indices = torch.arange(self.plan.total_amplitudes, device=state.device)
        for target in wires:
            mask = 1 << (self.plan.n_wires - int(target) - 1)
            signs = torch.where((indices & mask) == 0, 1.0, -1.0).to(
                device=state.device, dtype=probabilities.real.dtype
            )
            values.append((probabilities * signs.reshape(1, -1)).sum(dim=-1))
        return torch.stack(values, dim=-1)

    def summary(self) -> dict[str, Any]:
        plan_summary = self.plan.summary()
        jax_plan_summary = self.jax_plan.summary()
        is_sharded = (
            self.plan.world_size > 1
            and self.plan.distribution != "replicated_single_rank"
        )
        blockers = []
        if not is_sharded:
            blockers.append("world_size_is_one")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        blockers.extend(
            [
                "production_jax_pmap_executor_pending",
                "production_jax_collective_transport_pending",
            ]
        )
        rank_memory = tuple(shard.shard.local_state_bytes for shard in self.shards)
        topology_summary = (
            plan_summary["topology"].summary()
            if "topology" in plan_summary
            else self.plan.topology.summary()
        )
        memory_plan = _statevector_memory_plan_for_claimability(
            per_rank_shard_bytes=rank_memory,
            peak_buffer_bytes=int(plan_summary.get("peak_buffer_bytes", 0)),
            communication_buffer_count=int(
                plan_summary.get("communication_buffer_count", 0)
            ),
        )
        communication_plan = _statevector_communication_plan_for_claimability(
            topology_summary=topology_summary,
            communication_execution="jax_rank_local_amplitude_exchange",
            blockers=blockers,
        )
        payload = {
            "executor": "jax_sharded_statevector_executor",
            "claim_evidence_type": "development_smoke",
            "state_mode": "jax_sharded_statevector",
            "backend": "jax",
            "interface": "torch",
            "jax_backend": self.backend_policy.jax_backend,
            "torch_backend": self.backend_policy.torch_backend,
            "distributed_backend_policy": self.backend_policy.summary(),
            "distribution": self.plan.distribution,
            "distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "forward_distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "backward_distribution_semantics": "incomplete",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": bool(is_sharded),
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "forward_only_jax_sharded",
            "parameter_gradient_ready": False,
            "optimizer_update_semantics": "not_measured",
            "backward_uses_full_state_replay": False,
            "gradient_blockers": ("jax_sharded_statevector_backward_pending",),
            "world_size": self.plan.world_size,
            "local_world_size": self.plan.local_world_size,
            "node_count": self.plan.node_count,
            "n_wires": self.plan.n_wires,
            "batch_size": self.plan.bsz,
            "state_partition": self.plan.distribution,
            "sharded_wires": tuple(int(wire) for wire in self.plan.sharded_wires),
            "rank_coordinates": tuple(
                tuple(int(bit) for bit in coords)
                for coords in self.plan.topology.rank_coordinates
            ),
            "local_gate_count": self.local_gate_count,
            "distributed_gate_count": self.distributed_gate_count,
            "simulated_communication_count": self.simulated_communication_count,
            "simulated_communication_bytes": self.simulated_communication_bytes,
            "communication_execution": "jax_rank_local_amplitude_exchange",
            "communication_plan": communication_plan,
            "communication_tiers": topology_summary,
            "estimated_transfer_bytes": self.plan.estimated_transfer_bytes,
            "intra_node_communication_bytes": plan_summary[
                "intra_node_communication_bytes"
            ],
            "inter_node_communication_bytes": plan_summary[
                "inter_node_communication_bytes"
            ],
            "full_state_reconstruction_count": self.full_state_reconstruction_count,
            "local_memory_bytes_by_rank": rank_memory,
            "memory_plan": memory_plan,
            "rank_shards": tuple(shard.summary() for shard in self.shards),
            "jax_distributed_plan": jax_plan_summary,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_statevector_claimability(payload)
