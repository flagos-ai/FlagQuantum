# ruff: noqa: F401, F821
"""Tensor-network node, execution, and gradient result records."""

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


@dataclass(frozen=True)
class JAXTensorNetworkNode:
    """JAX tensor-network node with integer labels."""

    tensor: Any
    labels: tuple[int, ...]
    name: str = ""
    metadata: Mapping[str, Any] | None = None


@dataclass
class JAXShardedTensorNetworkResult:
    """Executable JAX tensor-network result with slices partitioned by rank."""

    rank_partials: tuple[JAXTNSliceRankState, ...]
    reduced_output: Any
    slicing: Any
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    n_wires: int
    bsz: int
    complex_bytes: int
    local_world_size: int
    node_count: int
    compute_backend: str = "local_simulated"
    compute_execution: str = "local_simulated_slice_compute"
    collective_backend: str = "local_simulated_psum"
    collective_execution: str = "local_simulated_psum"
    final_state_facade_count: int = 0
    _torch_state_cache: Any | None = None

    def state(self, *, refresh: bool = False) -> Any:
        """Return the reduced tensor-network output as a torch state facade."""

        if self._torch_state_cache is not None and not refresh:
            return self._torch_state_cache
        self._torch_state_cache = _jax_reduced_tn_output_to_torch_state(
            self.reduced_output,
            n_wires=self.n_wires,
            bsz=self.bsz,
            complex_bytes=self.complex_bytes,
        )
        self.final_state_facade_count += 1
        return self._torch_state_cache

    to_statevector = state
    wavefunction = state

    def probabilities(self) -> Any:
        torch = _require_torch()
        return torch.abs(self.state()) ** 2

    probability = probabilities

    def expectation_z(self, wire: int | None = None) -> Any:
        torch = _require_torch()
        state = self.state()
        probabilities = torch.abs(state) ** 2
        wires = tuple(range(self.n_wires)) if wire is None else (int(wire),)
        values = []
        indices = torch.arange(2**self.n_wires, device=state.device)
        for target in wires:
            mask = 1 << (self.n_wires - int(target) - 1)
            signs = torch.where((indices & mask) == 0, 1.0, -1.0).to(
                device=state.device, dtype=probabilities.real.dtype
            )
            values.append((probabilities * signs.reshape(1, -1)).sum(dim=-1))
        return torch.stack(values, dim=-1)

    def summary(self) -> dict[str, Any]:
        active_ranks = tuple(rank.rank for rank in self.rank_partials if rank.tasks)
        is_sharded = len(active_ranks) > 1 and int(self.slicing.n_slices) > 1
        root = active_ranks[0] if active_ranks else 0
        intra = 0
        inter = 0
        reduction_messages = []
        for rank_state in self.rank_partials:
            if rank_state.rank == root or not rank_state.tasks:
                continue
            transfer = _jax_array_nbytes(rank_state.partial)
            tier = _communication_tier(
                root, rank_state.rank, local_world_size=self.local_world_size
            )
            if tier == "intra_node":
                intra += transfer
            else:
                inter += transfer
            reduction_messages.append(
                {
                    "src_rank": rank_state.rank,
                    "dst_rank": root,
                    "tier": tier,
                    "estimated_transfer_bytes": int(transfer),
                    "collective": self.collective_execution,
                }
            )
        blockers = []
        if not is_sharded:
            blockers.append("insufficient_slice_parallelism")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        if self.compute_execution != "jax_pmap_slice_contraction":
            blockers.append("production_jax_pmap_tensor_network_compute_pending")
        else:
            blockers.append("node_tensor_inputs_replicated_for_slice_compute")
        if self.collective_execution != "jax_pmap_psum":
            blockers.append(
                "production_jax_tensor_network_collective_reduction_pending"
            )
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        payload = {
            "executor": "jax_sharded_tensor_network_executor",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_tensor_network",
            "backend": "jax",
            "interface": "torch",
            "jax_backend": self.backend_policy.jax_backend,
            "torch_backend": self.backend_policy.torch_backend,
            "distributed_backend_policy": self.backend_policy.summary(),
            "distribution_semantics": (
                "sharded_across_ranks"
                if is_sharded
                else "manual_sliced_tensor_contraction"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "forward_only_jax_sliced_tensor_network",
            "gradient_blockers": (
                "jax_sharded_tensor_network_reverse_contraction_pending",
            ),
            "compute_backend": self.compute_backend,
            "compute_execution": self.compute_execution,
            "collective_backend": self.collective_backend,
            "collective_execution": self.collective_execution,
            "world_size": len(self.rank_partials),
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.bsz,
            "sliced_labels": tuple(int(label) for label in self.slicing.sliced_labels),
            "slice_shape": tuple(int(dim) for dim in self.slicing.slice_shape),
            "slice_task_count": int(self.slicing.n_slices),
            "active_ranks": active_ranks,
            "rank_shards": tuple(rank.summary() for rank in self.rank_partials),
            "local_memory_bytes_by_rank": tuple(
                rank.summary()["partial_bytes"] for rank in self.rank_partials
            ),
            "communication_tiers": {
                "model": "jax_tensor_network_slice_reduction",
                "collective": self.collective_execution,
                "estimated_transfer_bytes": int(intra + inter),
                "intra_node_communication_bytes": int(intra),
                "inter_node_communication_bytes": int(inter),
                "reduction_messages": tuple(reduction_messages),
            },
            "intra_node_communication_bytes": int(intra),
            "inter_node_communication_bytes": int(inter),
            "peak_intermediate_elements_per_slice": int(self.slicing.peak_size),
            "partial_output_bytes": _jax_array_nbytes(self.reduced_output),
            "final_state_facade_count": self.final_state_facade_count,
            "silent_statevector_fallback": False,
            "full_tensor_network_fallback_count": 0,
            "jax_distributed_plan": self.jax_plan.summary(),
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_distributed_evidence_contract(payload)


@dataclass
class JAXSlicedTensorNetworkGradientResult:
    """Reverse-mode result for a sliced tensor-network contraction."""

    value: Any
    node_gradients: tuple[Any, ...]
    slicing: Any
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    n_wires: int
    bsz: int
    complex_bytes: int
    local_world_size: int
    node_count: int
    compute_backend: str
    compute_execution: str
    collective_backend: str
    collective_execution: str
    observable: str

    def torch_value(self) -> Any:
        import numpy as np

        torch = _require_torch()
        real_dtype = torch.float64 if self.complex_bytes == 16 else torch.float32
        return torch.as_tensor(
            np.asarray(self.value).copy(), dtype=real_dtype, device=torch.device("cpu")
        )

    def torch_node_gradients(self) -> tuple[Any, ...]:
        import numpy as np

        torch = _require_torch()
        dtype = _torch_complex_dtype(self.complex_bytes)
        return tuple(
            torch.as_tensor(
                np.asarray(grad).copy(), dtype=dtype, device=torch.device("cpu")
            )
            for grad in self.node_gradients
        )

    def summary(self) -> dict[str, Any]:
        tasks_by_rank = _tasks_by_rank_from_slicing(
            self.slicing, self.jax_plan.world_size
        )
        active = tuple(rank for rank, count in enumerate(tasks_by_rank) if count > 0)
        is_sharded = len(active) > 1 and int(self.slicing.n_slices) > 1
        output_bytes = (
            int(self.bsz) * (2 ** int(self.n_wires)) * int(self.complex_bytes)
        )
        local_memory = tuple(int(count * output_bytes) for count in tasks_by_rank)
        root = active[0] if active else 0
        estimated_transfer = sum(local_memory[rank] for rank in active if rank != root)
        blockers = []
        if not is_sharded:
            blockers.append("insufficient_slice_parallelism")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        if self.compute_execution != "jax_pmap_slice_contraction":
            blockers.append("production_jax_pmap_tensor_network_compute_pending")
        if self.collective_execution != "jax_pmap_psum":
            blockers.append(
                "production_jax_tensor_network_collective_reduction_pending"
            )
        blockers.append("parameter_gate_pullback_pending")
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        payload = {
            "executor": "jax_sliced_tensor_network_reverse_mode",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_tensor_network",
            "backend": "jax",
            "interface": "torch",
            "observable": self.observable,
            "distribution_semantics": (
                "sharded_across_ranks"
                if is_sharded
                else "manual_sliced_tensor_contraction"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "jax_reverse_mode_sliced_contraction",
            "gradient_target": "tensor_network_node_tensors",
            "gradient_blockers": ("parameter_gate_pullback_pending",),
            "compute_backend": self.compute_backend,
            "compute_execution": self.compute_execution,
            "collective_backend": self.collective_backend,
            "collective_execution": self.collective_execution,
            "world_size": self.jax_plan.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.bsz,
            "sliced_labels": tuple(int(label) for label in self.slicing.sliced_labels),
            "slice_shape": tuple(int(dim) for dim in self.slicing.slice_shape),
            "slice_task_count": int(self.slicing.n_slices),
            "active_ranks": active,
            "rank_shards": tuple(
                {
                    "rank": rank,
                    "state_partition": "tensor_network_slices",
                    "slice_task_count": int(tasks_by_rank[rank]),
                    "local_memory_bytes": local_memory[rank],
                }
                for rank in range(self.jax_plan.world_size)
            ),
            "local_memory_bytes_by_rank": local_memory,
            "communication_tiers": {
                "model": "jax_tensor_network_reverse_slice_reduction",
                "collective": self.collective_execution,
                "estimated_transfer_bytes": int(estimated_transfer),
            },
            "estimated_transfer_bytes": int(estimated_transfer),
            "node_gradient_count": len(self.node_gradients),
            "node_gradient_shapes": tuple(
                tuple(int(dim) for dim in grad.shape) for grad in self.node_gradients
            ),
            "jax_distributed_plan": self.jax_plan.summary(),
        }
        return _attach_distributed_evidence_contract(payload)


@dataclass
class JAXSlicedTensorNetworkParameterGradientResult:
    """Reverse-mode sliced TN value/gradient with pullback to circuit parameters."""

    value: Any
    gradient: Any
    slicing: Any
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    n_wires: int
    bsz: int
    complex_bytes: int
    parameter_shape: tuple[int, ...]
    local_world_size: int
    node_count: int
    compute_backend: str
    compute_execution: str
    collective_backend: str
    collective_execution: str
    observable: str

    def torch_value(self, *, like: Any | None = None) -> Any:
        import numpy as np

        torch = _require_torch()
        dtype = torch.float64 if self.complex_bytes == 16 else torch.float32
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
        dtype = torch.float64 if self.complex_bytes == 16 else torch.float32
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

    def summary(self) -> dict[str, Any]:
        tasks_by_rank = _tasks_by_rank_from_slicing(
            self.slicing, self.jax_plan.world_size
        )
        active = tuple(rank for rank, count in enumerate(tasks_by_rank) if count > 0)
        is_sharded = len(active) > 1 and int(self.slicing.n_slices) > 1
        output_bytes = (
            int(self.bsz) * (2 ** int(self.n_wires)) * int(self.complex_bytes)
        )
        local_memory = tuple(int(count * output_bytes) for count in tasks_by_rank)
        root = active[0] if active else 0
        estimated_transfer = sum(local_memory[rank] for rank in active if rank != root)
        blockers = []
        if not is_sharded:
            blockers.append("insufficient_slice_parallelism")
        if self.backend_policy.profile != "production":
            blockers.append("single_process_development_simulator")
        if self.compute_execution != "jax_pmap_slice_contraction":
            blockers.append("production_jax_pmap_tensor_network_compute_pending")
        else:
            blockers.append("node_tensor_inputs_replicated_for_slice_compute")
        if self.collective_execution != "jax_pmap_psum":
            blockers.append(
                "production_jax_tensor_network_collective_reduction_pending"
            )
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        payload = {
            "executor": "jax_sliced_tensor_network_parameter_reverse_mode",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_tensor_network",
            "backend": "jax",
            "interface": "torch",
            "observable": self.observable,
            "distribution_semantics": (
                "sharded_across_ranks"
                if is_sharded
                else "manual_sliced_tensor_contraction"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "jax_reverse_mode_sliced_parameter_contraction",
            "gradient_target": "parameterized_gate_tensors",
            "gradient_blockers": tuple(dict.fromkeys(blockers)),
            "compute_backend": self.compute_backend,
            "compute_execution": self.compute_execution,
            "collective_backend": self.collective_backend,
            "collective_execution": self.collective_execution,
            "world_size": self.jax_plan.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.n_wires,
            "batch_size": self.bsz,
            "parameter_shape": self.parameter_shape,
            "sliced_labels": tuple(int(label) for label in self.slicing.sliced_labels),
            "slice_shape": tuple(int(dim) for dim in self.slicing.slice_shape),
            "slice_task_count": int(self.slicing.n_slices),
            "active_ranks": active,
            "rank_shards": tuple(
                {
                    "rank": rank,
                    "state_partition": "tensor_network_slices",
                    "slice_task_count": int(tasks_by_rank[rank]),
                    "local_memory_bytes": local_memory[rank],
                }
                for rank in range(self.jax_plan.world_size)
            ),
            "local_memory_bytes_by_rank": local_memory,
            "communication_tiers": {
                "model": "jax_tensor_network_parameter_reverse_slice_reduction",
                "collective": self.collective_execution,
                "estimated_transfer_bytes": int(estimated_transfer),
            },
            "estimated_transfer_bytes": int(estimated_transfer),
            "jax_distributed_plan": self.jax_plan.summary(),
        }
        return _attach_distributed_evidence_contract(payload)
