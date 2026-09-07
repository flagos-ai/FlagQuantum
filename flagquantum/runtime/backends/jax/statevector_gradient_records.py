"""Statevector parameter-gradient records and shard initialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from ....core.ir import CircuitIR
from ....simulation.jax.statevector import jax_initial_statevector_shard
from ...distributed.backend_policy import DistributedBackendPolicy
from .common import node_count as _node_count
from .planning_core import JAXDistributedQuantumPlan
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)
from .runtime_environment import _jnp_device_put, _require_jax, _require_torch
from .statevector.records import (
    JAXStatevectorShardState,
    _statevector_communication_plan_for_claimability,
    _statevector_memory_plan_for_claimability,
)


@dataclass
class JAXShardedStatevectorParameterGradientResult:
    """Amplitude-sharded statevector value/gradient with parameter pullback."""

    value: Any
    gradient: Any
    plan: Any
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    parameter_shape: tuple[int, ...]
    local_gate_count: int
    distributed_gate_count: int
    simulated_communication_count: int
    simulated_communication_bytes: int
    observable: str
    backward_backend: str = "local_simulated"
    backward_execution: str = "local_simulated_backward"
    production_device_summary: Mapping[str, Any] | None = None
    production_blockers: tuple[str, ...] = ()
    backward_strategy: str = "reverse_mode_amplitude_sharded"
    parameter_ownership_semantics: str = "not_measured"
    gradient_ownership_semantics: str = "not_measured"
    optimizer_update_semantics: str = "not_measured"
    optimizer_update_ownership_semantics: str = "not_measured"
    training_step_count: int = 0
    parameter_ownership: tuple[Mapping[str, Any], ...] = ()
    gradient_ownership: tuple[Mapping[str, Any], ...] = ()
    optimizer_update_ownership: tuple[Mapping[str, Any], ...] = ()
    optimizer_step_evidence: Mapping[str, Any] | None = None

    def torch_value(self, *, like: Any | None = None) -> Any:
        import numpy as np

        torch = _require_torch()
        dtype = torch.float64 if self.plan.complex_bytes == 16 else torch.float32
        device = torch.device("cpu")
        if like is not None and hasattr(like, "device"):
            device = like.device
            dtype = (
                like.dtype
                if getattr(like, "dtype", None) in {torch.float32, torch.float64}
                else dtype
            )
        return torch.as_tensor(
            np.asarray(self.value).copy(), dtype=dtype, device=device
        )

    def torch_gradient(self, *, like: Any | None = None) -> Any:
        import numpy as np

        torch = _require_torch()
        dtype = torch.float64 if self.plan.complex_bytes == 16 else torch.float32
        device = torch.device("cpu")
        if like is not None and hasattr(like, "device"):
            device = like.device
            dtype = (
                like.dtype
                if getattr(like, "dtype", None) in {torch.float32, torch.float64}
                else dtype
            )
        return torch.as_tensor(
            np.asarray(self.gradient).copy(), dtype=dtype, device=device
        )

    def with_sharded_optimizer_step_evidence(
        self,
        *,
        training_step_count: int = 1,
        optimizer_name: str = "explicit_sharded_optimizer_step",
    ) -> "JAXShardedStatevectorParameterGradientResult":
        """Return a copy with explicit sharded optimizer-step ownership evidence."""

        production_backward_executions = {"jax_pmap_backward", "jax_shard_map_backward"}
        is_sharded = (
            self.plan.world_size > 1
            and self.plan.distribution != "replicated_single_rank"
        )
        if not (
            is_sharded
            and self.backend_policy.profile == "production"
            and self.backward_execution in production_backward_executions
            and not self.production_blockers
        ):
            raise ValueError(
                "sharded optimizer evidence requires executed production sharded value-and-grad"
            )
        try:
            steps = int(training_step_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("training_step_count must be a positive integer") from exc
        if steps <= 0:
            raise ValueError("training_step_count must be a positive integer")

        parameter_count = 1
        for dim in self.parameter_shape:
            parameter_count *= max(1, int(dim))
        ownership = []
        for rank in range(int(self.plan.world_size)):
            start = rank * parameter_count // int(self.plan.world_size)
            end = (rank + 1) * parameter_count // int(self.plan.world_size)
            ownership.append(
                {
                    "rank": rank,
                    "parameter_start": start,
                    "parameter_end": end,
                    "owned_parameter_count": end - start,
                    "ownership": "parameter_gradient_optimizer_slice",
                }
            )
        parameter_ownership = tuple(
            dict(item, tensor="parameters") for item in ownership
        )
        gradient_ownership = tuple(dict(item, tensor="gradients") for item in ownership)
        optimizer_update_ownership = tuple(
            dict(item, tensor="optimizer_updates") for item in ownership
        )
        evidence = {
            "optimizer_name": str(optimizer_name),
            "training_step_count": steps,
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "parameter_ownership": parameter_ownership,
            "gradient_ownership": gradient_ownership,
            "optimizer_update_ownership": optimizer_update_ownership,
        }
        return replace(
            self,
            parameter_ownership_semantics="sharded_across_ranks",
            gradient_ownership_semantics="sharded_across_ranks",
            optimizer_update_semantics="sharded_across_ranks",
            optimizer_update_ownership_semantics="sharded_across_ranks",
            training_step_count=steps,
            parameter_ownership=parameter_ownership,
            gradient_ownership=gradient_ownership,
            optimizer_update_ownership=optimizer_update_ownership,
            optimizer_step_evidence=evidence,
        )

    def summary(self) -> dict[str, Any]:
        plan_summary = self.plan.summary()
        jax_plan_summary = dict(self.jax_plan.summary())
        is_sharded = (
            self.plan.world_size > 1
            and self.plan.distribution != "replicated_single_rank"
        )
        production_backward_executions = {"jax_pmap_backward", "jax_shard_map_backward"}
        value_gradient_ready = (
            is_sharded
            and self.backend_policy.profile == "production"
            and self.backward_execution in production_backward_executions
            and not self.production_blockers
        )
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        blockers = []
        if not is_sharded:
            blockers.append("world_size_is_one")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        if self.backward_execution not in production_backward_executions:
            blockers.extend(
                (
                    "production_jax_statevector_backward_pending",
                    "production_jax_collective_transport_pending",
                )
            )
        blockers.extend(self.production_blockers)
        if (
            self.backward_execution in production_backward_executions
            and not self.production_blockers
        ):
            communication_tiers = dict(jax_plan_summary.get("communication_tiers", {}))
            communication_tiers["model"] = (
                "jax_pmap_collectives_executed_from_statevector_topology"
                if self.backward_execution == "jax_pmap_backward"
                else "jax_shard_map_mesh_collectives_executed_from_statevector_topology"
            )
            jax_plan_summary.update(
                {
                    "status": "executed_by_runtime",
                    "runtime_backward_backend": self.backward_backend,
                    "runtime_backward_execution": self.backward_execution,
                    "scalability_claim_allowed": False,
                    "claim_evidence_type": evidence_type,
                    "release_gate_allowed": False,
                    "production_value_gradient_ready": value_gradient_ready,
                    "scalability_blockers": tuple(dict.fromkeys(blockers)),
                    "gradient_blockers": tuple(dict.fromkeys(blockers)),
                    "communication_tiers": communication_tiers,
                }
            )
        rank_memory = tuple(shard.local_state_bytes for shard in self.plan.shards)
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
        communication_execution = (
            "jax_pmap_amplitude_exchange_backward_replay"
            if self.backward_execution == "jax_pmap_backward"
            else (
                "jax_shard_map_mesh_amplitude_exchange_backward_replay"
                if self.backward_execution == "jax_shard_map_backward"
                else "jax_local_simulated_amplitude_exchange_backward_replay"
            )
        )
        communication_plan = _statevector_communication_plan_for_claimability(
            topology_summary=topology_summary,
            communication_execution=communication_execution,
            blockers=blockers,
        )
        payload = {
            "executor": "jax_sharded_statevector_parameter_reverse_mode",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_statevector",
            "backend": "jax",
            "interface": "torch",
            "observable": self.observable,
            "distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "forward_distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "backward_distribution_semantics": (
                "sharded_across_ranks" if value_gradient_ready else "incomplete"
            ),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": bool(is_sharded),
            "production_value_gradient_ready": value_gradient_ready,
            "parameter_gradient_ready": value_gradient_ready,
            "single_gpu_expected_oom": False,
            "capacity_baseline_device": "not_measured",
            "capacity_failure_reason": "not_measured",
            "parameter_ownership_semantics": self.parameter_ownership_semantics,
            "gradient_ownership_semantics": self.gradient_ownership_semantics,
            "optimizer_update_semantics": self.optimizer_update_semantics,
            "optimizer_update_ownership_semantics": self.optimizer_update_ownership_semantics,
            "training_step_count": int(self.training_step_count),
            "parameter_ownership": self.parameter_ownership,
            "gradient_ownership": self.gradient_ownership,
            "optimizer_update_ownership": self.optimizer_update_ownership,
            "optimizer_step_evidence": dict(self.optimizer_step_evidence or {}),
            "gradient_distribution_semantics": (
                "sharded_across_ranks" if value_gradient_ready else "incomplete"
            ),
            "backward_uses_full_state_replay": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "jax_reverse_mode_amplitude_sharded_parameter_backprop",
            "gradient_target": "parameterized_gate_tensors",
            "gradient_blockers": tuple(dict.fromkeys(blockers)),
            "backward_backend": self.backward_backend,
            "backward_execution": self.backward_execution,
            "production_device_summary": dict(self.production_device_summary or {}),
            "production_blockers": tuple(self.production_blockers),
            "backward_strategy": self.backward_strategy,
            "backend_policy": self.backend_policy.summary(),
            "parameter_shape": self.parameter_shape,
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
            "statevector_gate_communication_count": self.distributed_gate_count,
            "gradient_reduction_collective_count": (
                1 if self.backward_execution in production_backward_executions else 0
            ),
            "communication_execution": communication_execution,
            "communication_plan": communication_plan,
            "communication_tiers": topology_summary,
            "estimated_transfer_bytes": self.plan.estimated_transfer_bytes,
            "intra_node_communication_bytes": plan_summary[
                "intra_node_communication_bytes"
            ],
            "inter_node_communication_bytes": plan_summary[
                "inter_node_communication_bytes"
            ],
            "local_memory_bytes_by_rank": rank_memory,
            "memory_plan": memory_plan,
            "rank_shards": tuple(
                {
                    "rank": shard.rank,
                    "amplitude_start": shard.amplitude_start,
                    "amplitude_end": shard.amplitude_end,
                    "local_amplitudes": shard.local_amplitudes,
                    "local_state_bytes": shard.local_state_bytes,
                    "state_partition": self.plan.distribution,
                    "rank_coordinates": tuple(
                        int(bit)
                        for bit in self.plan.topology.rank_coordinates[int(shard.rank)]
                    ),
                    "basis_filter": tuple(
                        {
                            "wire": int(wire),
                            "bit": int(bit),
                        }
                        for wire, bit in zip(
                            self.plan.sharded_wires,
                            self.plan.topology.rank_coordinates[int(shard.rank)],
                        )
                    ),
                }
                for shard in self.plan.shards
            ),
            "full_state_reconstruction_count": 0,
            "silent_statevector_fallback": False,
            "jax_distributed_plan": jax_plan_summary,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_statevector_claimability(payload)


def _statevector_plan(
    ir: CircuitIR,
    *,
    policy: DistributedBackendPolicy,
    world_size: int,
    local_world_size: int,
    bsz: int,
    complex_bytes: int,
) -> JAXDistributedQuantumPlan:
    from ..statevector.planning import plan_distributed_statevector

    state_plan = plan_distributed_statevector(
        ir,
        world_size=world_size,
        local_world_size=local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    state_summary = state_plan.summary()
    rank_ownership = tuple(
        {
            "rank": item["rank"],
            "state_partition": "amplitude_range",
            "amplitude_start": item["amplitude_start"],
            "amplitude_end": item["amplitude_end"],
            "local_amplitudes": item["local_amplitudes"],
            "local_memory_bytes": item["local_state_bytes"],
        }
        for item in state_summary["rank_shards"]
    )
    local_memory = tuple(int(item["local_memory_bytes"]) for item in rank_ownership)
    blockers = (
        "jax_pmap_statevector_executor_pending",
        "rank_local_jax_kernel_is_not_capacity_scaling",
    )
    gradient_blockers = ("jax_sharded_statevector_backward_pending",)
    return JAXDistributedQuantumPlan(
        mode="statevector",
        n_wires=ir.n_wires,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=_node_count(world_size, local_world_size),
        backend_policy=policy,
        rank_ownership=rank_ownership,
        communication_tiers={
            "model": "jax_pmap_collectives_planned_from_statevector_topology",
            "topology_layout": state_summary["topology_layout"],
            "estimated_transfer_bytes": state_summary["estimated_transfer_bytes"],
            "intra_node_communication_bytes": state_summary[
                "intra_node_communication_bytes"
            ],
            "inter_node_communication_bytes": state_summary[
                "inter_node_communication_bytes"
            ],
            "communication_segment_count": state_summary["communication_segment_count"],
        },
        local_memory_bytes_by_rank=local_memory,
        blockers=blockers,
        gradient_blockers=gradient_blockers,
        task_summary={
            "sharded_wires": state_summary["sharded_wires"],
            "communication_gate_count": state_summary["communication_gate_count"],
            "execution_segment_count": state_summary["execution_segment_count"],
        },
    )


def _initialize_jax_statevector_shard(
    plan: Any, *, rank: int, dtype: Any, device: Any | None
) -> JAXStatevectorShardState:
    from ..statevector.local_execution import _rank_global_indices, _shard_by_rank

    jax, jnp = _require_jax()
    torch = _require_torch()
    shard = _shard_by_rank(plan.shards)[int(rank)]
    indices_torch = _rank_global_indices(plan, int(rank), device=torch.device("cpu"))
    indices_tuple = tuple(int(item) for item in indices_torch.detach().cpu().tolist())
    if int(plan.total_amplitudes) > 2**31:
        jax.config.update("jax_enable_x64", True)
        index_dtype = jnp.int64
    else:
        index_dtype = jnp.int32
    global_indices = _jnp_device_put(
        jnp.asarray(indices_tuple, dtype=index_dtype), device
    )
    amplitudes = jax_initial_statevector_shard(
        global_indices,
        batch_size=int(plan.bsz),
        dtype=dtype,
    )
    amplitudes = _jnp_device_put(amplitudes, device)
    return JAXStatevectorShardState(
        rank=int(rank),
        shard=shard,
        amplitudes=amplitudes,
        global_indices=global_indices,
        global_indices_tuple=indices_tuple,
    )
