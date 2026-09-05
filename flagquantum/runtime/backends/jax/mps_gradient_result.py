# ruff: noqa: F401, F821
"""Sharded MPS parameter-gradient execution result."""

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
from .mps_evidence import _build_mps_backward_resource_evidence
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)


@dataclass
class JAXShardedMPSParameterGradientResult:
    """Site-sharded MPS value/gradient with parameter pullback."""

    value: Any
    gradient: Any
    rank_shards: tuple[JAXMPSRankShardState, ...]
    shard_plans: tuple[Any, ...]
    jax_plan: JAXDistributedQuantumPlan
    backend_policy: DistributedBackendPolicy
    n_wires: int
    bsz: int
    complex_bytes: int
    parameter_shape: tuple[int, ...]
    max_bond: int | None
    cutoff: float
    owned_instruction_count: int
    sharded_kernel_count: int
    boundary_sync_count: int
    boundary_transfer_bytes: int
    truncation_records: tuple[Mapping[str, Any], ...]
    boundary_protocols: tuple[Mapping[str, Any], ...]
    boundary_adjoint_exchange_evidence: Mapping[str, Any]
    parameter_gradient_ownership_evidence: Mapping[str, Any]
    observable: str
    backward_backend: str = "local_simulated"
    backward_execution: str = "local_simulated_backward"
    production_device_summary: Mapping[str, Any] | None = None
    production_blockers: tuple[str, ...] = ()
    backward_strategy: str = "reverse_mode_site_sharded_transfer"
    parameters: Any | None = None
    updated_parameters: Any | None = None
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

    def with_sharded_optimizer_step_evidence(
        self,
        *,
        learning_rate: float = 0.01,
        optimizer_name: str = "owner_local_sgd",
    ) -> "JAXShardedMPSParameterGradientResult":
        """Execute one owner-local SGD step and attach development evidence."""

        import numpy as np

        if len(self.rank_shards) <= 1:
            raise ValueError("sharded MPS optimizer evidence requires world_size > 1")
        if self.parameters is None:
            raise ValueError(
                "sharded MPS optimizer evidence requires captured parameters"
            )
        ownership_evidence = dict(self.parameter_gradient_ownership_evidence)
        if not bool(ownership_evidence.get("valid", False)):
            raise ValueError(
                "sharded MPS optimizer evidence requires complete parameter-gradient ownership"
            )

        parameter_values = np.asarray(self.parameters)
        gradient_values = np.asarray(self.gradient)
        if parameter_values.shape != gradient_values.shape:
            raise ValueError("parameter and gradient shapes must match")
        parameter_flat = parameter_values.reshape(-1)
        gradient_flat = gradient_values.reshape(-1)
        if parameter_flat.size == 0:
            raise ValueError(
                "sharded MPS optimizer evidence requires nonempty parameters"
            )
        try:
            step_size = float(learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("learning_rate must be a positive finite number") from exc
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("learning_rate must be a positive finite number")

        ownership_by_index: dict[int, Mapping[str, Any]] = {}
        for record in ownership_evidence.get("parameter_gradient_ownership", ()):
            if not isinstance(record, Mapping):
                raise ValueError(
                    "parameter-gradient ownership records must be mappings"
                )
            flat_index = int(record.get("parameter_flat_index", -1))
            owner_rank = int(record.get("owner_rank", -1))
            if (
                flat_index < 0
                or flat_index >= parameter_flat.size
                or flat_index in ownership_by_index
                or owner_rank < 0
                or owner_rank >= len(self.rank_shards)
            ):
                raise ValueError(
                    "parameter-gradient ownership must cover each parameter exactly once"
                )
            ownership_by_index[flat_index] = record
        if set(ownership_by_index) != set(range(int(parameter_flat.size))):
            raise ValueError(
                "parameter-gradient ownership must cover each parameter exactly once"
            )

        updated_flat = parameter_flat.copy()
        parameter_ownership = []
        gradient_ownership = []
        optimizer_update_ownership = []
        update_records = []
        for flat_index in range(int(parameter_flat.size)):
            source = ownership_by_index[flat_index]
            owner_rank = int(source["owner_rank"])
            parameter_id = str(source.get("parameter_id", f"parameter[{flat_index}]"))
            parameter_index = tuple(
                int(index)
                for index in source.get(
                    "parameter_index",
                    np.unravel_index(flat_index, parameter_values.shape),
                )
            )
            parameter_value = float(parameter_flat[flat_index])
            gradient_value = float(gradient_flat[flat_index])
            update_value = -step_size * gradient_value
            updated_value = parameter_value + update_value
            updated_flat[flat_index] = updated_value
            common = {
                "parameter_id": parameter_id,
                "parameter_flat_index": flat_index,
                "parameter_index": parameter_index,
                "owner_rank": owner_rank,
                "rank": owner_rank,
                "site_range": tuple(source.get("site_range", ())),
            }
            parameter_ownership.append(
                {
                    **common,
                    "parameter_owner_rank": owner_rank,
                    "parameter_value_before": parameter_value,
                    "ownership_semantics": "rank_owned_parameter",
                }
            )
            gradient_ownership.append(
                {
                    **common,
                    "gradient_owner_rank": owner_rank,
                    "gradient_value": gradient_value,
                    "ownership_semantics": "rank_owned_gradient",
                }
            )
            update_record = {
                **common,
                "update_owner_rank": owner_rank,
                "gradient_owner_rank": owner_rank,
                "parameter_owner_rank": owner_rank,
                "writeback_route": "rank_local_parameter_owner_writeback",
                "optimizer_update_value": update_value,
                "parameter_value_after": updated_value,
                "ownership_semantics": "rank_owned_optimizer_update",
            }
            optimizer_update_ownership.append(update_record)
            update_records.append(update_record)

        parameter_ownership_out = tuple(parameter_ownership)
        gradient_ownership_out = tuple(gradient_ownership)
        optimizer_update_ownership_out = tuple(optimizer_update_ownership)
        evidence = {
            "status": "local_cpu_executed",
            "valid": True,
            "execution_scope": "development_cpu",
            "evidence_scope": "mps_sharded_optimizer_ownership_step",
            "claim_evidence_type": "development_smoke",
            "optimizer_name": str(optimizer_name),
            "learning_rate": step_size,
            "training_step_count": 1,
            "parameter_count": int(parameter_flat.size),
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "parameter_ownership": parameter_ownership_out,
            "gradient_ownership": gradient_ownership_out,
            "optimizer_update_ownership": optimizer_update_ownership_out,
            "records": tuple(update_records),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": (
                "mps_optimizer_step_development_cpu_only",
                "mps_production_sharded_optimizer_execution_pending",
            ),
        }
        return replace(
            self,
            updated_parameters=updated_flat.reshape(parameter_values.shape),
            parameter_ownership_semantics="sharded_across_ranks",
            gradient_ownership_semantics="sharded_across_ranks",
            optimizer_update_semantics="sharded_across_ranks",
            optimizer_update_ownership_semantics="sharded_across_ranks",
            training_step_count=1,
            parameter_ownership=parameter_ownership_out,
            gradient_ownership=gradient_ownership_out,
            optimizer_update_ownership=optimizer_update_ownership_out,
            optimizer_step_evidence=evidence,
        )

    def summary(self) -> dict[str, Any]:
        is_sharded = len(self.rank_shards) > 1
        local_memory = tuple(
            int(shard.summary()["local_tensor_bytes"]) for shard in self.rank_shards
        )
        jax_plan_summary = self.jax_plan.summary()
        bond_ownership = tuple(
            jax_plan_summary.get("communication_tiers", {}).get("boundary_edges", ())
        )
        exchange_evidence = dict(self.boundary_adjoint_exchange_evidence)
        exchange_records = tuple(exchange_evidence.get("records", ()))
        exchange_routes = tuple(exchange_evidence.get("boundary_gradient_routes", ()))
        exchange_ownership = tuple(
            exchange_evidence.get("boundary_gradient_ownership", ())
        )
        exchange_blockers = tuple(exchange_evidence.get("blockers", ()))
        exchange_bytes = int(exchange_evidence.get("communication_bytes", 0) or 0)
        ownership_evidence = dict(self.parameter_gradient_ownership_evidence)
        parameter_gradient_ownership = tuple(
            ownership_evidence.get("parameter_gradient_ownership", ())
        )
        ownership_blockers = tuple(ownership_evidence.get("blockers", ()))
        ownership_memory = tuple(
            ownership_evidence.get("local_gradient_bytes_by_rank", ())
        )
        resource_evidence = _build_mps_backward_resource_evidence(
            tuple(shard.summary() for shard in self.rank_shards),
            self.boundary_protocols,
            exchange_records,
            ownership_memory,
            self.truncation_records,
            world_size=len(self.rank_shards),
            local_world_size=self.jax_plan.local_world_size,
            node_count=self.jax_plan.node_count,
            execution_scope="development_cpu",
            communication_executed=True,
        )
        backward_memory_plan = dict(resource_evidence["mps_backward_memory_plan"])
        backward_communication_plan = dict(
            resource_evidence["mps_backward_communication_plan"]
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
        if self.backward_execution != "jax_pmap_backward":
            blockers.extend(
                (
                    "production_jax_pmap_mps_backward_pending",
                    "production_jax_mps_boundary_transport_pending",
                )
            )
        blockers.extend(self.production_blockers)
        blockers.extend(exchange_blockers)
        blockers.extend(ownership_blockers)
        blockers.extend(resource_evidence.get("blockers", ()))
        if self.truncation_records:
            blockers.append("truncated_svd_custom_pullback_validation_pending")
        evidence_type = (
            "development_smoke"
            if self.backend_policy.profile == "development"
            else "production_runtime"
        )
        payload = {
            "executor": "jax_sharded_mps_parameter_reverse_mode",
            "claim_evidence_type": evidence_type,
            "state_mode": "jax_sharded_mps",
            "backend": "jax",
            "interface": "torch",
            "observable": self.observable,
            "distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": tuple(dict.fromkeys(blockers)),
            "gradient_execution": "jax_reverse_mode_site_sharded_mps_parameter_backprop",
            "gradient_target": "parameterized_gate_tensors",
            "gradient_blockers": tuple(dict.fromkeys(blockers)),
            "mps_forward_distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "mps_backward_distribution_semantics": (
                "sharded_across_ranks"
                if is_sharded
                and self.backward_execution
                in {"jax_pmap_backward", "jax_shard_map_backward"}
                else (
                    "local_simulation"
                    if self.backward_execution == "local_simulated_backward"
                    else "incomplete"
                )
            ),
            "site_shard_ownership": tuple(
                shard.summary() for shard in self.rank_shards
            ),
            "bond_shard_ownership": bond_ownership,
            "gradient_distribution_semantics": ownership_evidence.get(
                "gradient_distribution_semantics", "unknown"
            ),
            "parameter_gradient_ownership": (
                parameter_gradient_ownership
                if parameter_gradient_ownership
                else "unknown"
            ),
            "parameter_gradient_ownership_evidence": ownership_evidence,
            "boundary_gradient_ownership": (
                exchange_ownership
                if exchange_ownership
                else "not_required" if not self.boundary_protocols else "unknown"
            ),
            "boundary_adjoint_exchange": exchange_evidence,
            "boundary_gradient_routes": (
                exchange_routes
                if exchange_routes
                else "not_required" if not self.boundary_protocols else ()
            ),
            "boundary_adjoint_exchange_evidence": exchange_evidence,
            "boundary_adjoint_exchange_records": exchange_records,
            "canonicalization_backward_strategy": {
                "status": "pending",
                "strategy": "cross_shard_canonicalization_pullback",
            },
            "truncation_gradient_metadata": (
                {"status": "pending", "records": self.truncation_records}
                if self.truncation_records
                else {"status": "not_required", "cutoff": self.cutoff}
            ),
            "mps_backward_resource_evidence": resource_evidence,
            "mps_backward_memory_plan": backward_memory_plan,
            "mps_backward_communication_plan": {
                **backward_communication_plan,
                "boundary_adjoint_exchange": exchange_evidence,
                "boundary_gradient_routes": (
                    exchange_routes
                    if exchange_routes
                    else "not_required" if not self.boundary_protocols else ()
                ),
            },
            "parameter_ownership_semantics": self.parameter_ownership_semantics,
            "gradient_ownership_semantics": self.gradient_ownership_semantics,
            "optimizer_update_semantics": self.optimizer_update_semantics,
            "optimizer_update_ownership_semantics": self.optimizer_update_ownership_semantics,
            "training_step_count": self.training_step_count,
            "parameter_ownership": self.parameter_ownership,
            "gradient_ownership": self.gradient_ownership,
            "optimizer_update_ownership": self.optimizer_update_ownership,
            "optimizer_step_evidence": dict(self.optimizer_step_evidence or {}),
            "fallback_semantics": (
                "local_simulation"
                if self.backward_execution == "local_simulated_backward"
                else "none"
            ),
            "backward_backend": self.backward_backend,
            "backward_execution": self.backward_execution,
            "production_device_summary": dict(self.production_device_summary or {}),
            "production_blockers": tuple(self.production_blockers),
            "backward_strategy": self.backward_strategy,
            "parameter_shape": self.parameter_shape,
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
            "boundary_transfer_bytes": int(self.boundary_transfer_bytes),
            "boundary_adjoint_exchange_bytes": exchange_bytes,
            "communication_tiers": {
                "model": "jax_mps_boundary_adjoint_exchange_probe",
                "boundary_edge_count": len(self.boundary_protocols),
                "estimated_transfer_bytes": int(self.boundary_transfer_bytes),
                "intra_node_communication_bytes": int(intra),
                "inter_node_communication_bytes": int(inter),
                "boundary_protocols": self.boundary_protocols,
                "boundary_adjoint_exchange_evidence": exchange_evidence,
                "mps_backward_resource_evidence": resource_evidence,
            },
            "intra_node_communication_bytes": int(intra),
            "inter_node_communication_bytes": int(inter),
            "local_memory_bytes_by_rank": local_memory,
            "rank_shards": tuple(shard.summary() for shard in self.rank_shards),
            "truncation_steps": len(self.truncation_records),
            "truncation_records": self.truncation_records,
            "full_mps_reconstruction_count": 0,
            "statevector_facade_count": 0,
            "silent_statevector_fallback": False,
            "full_state_fallback_count": 0,
            "jax_distributed_plan": jax_plan_summary,
            "rank_local_jax_kernel_allowed_for_capacity_claim": False,
        }
        return _attach_mps_backward_readiness(payload)
