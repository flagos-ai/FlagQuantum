# ruff: noqa: F401, F821
"""MPS canonicalization, truncation, optimizer pullbacks, and protocol."""

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


def _execute_minimal_mps_canonicalization_truncation_pullback(
    parameter: Any,
    *,
    truncation_mode: str = "exact",
    cutoff: float = 0.0,
) -> dict[str, Any]:
    """Execute constrained QR/SVD pullback evidence for a boundary MPS bond."""

    import numpy as np

    jax, jnp = _require_jax()
    jax.config.update("jax_enable_x64", True)
    parameter_values = np.asarray(parameter)
    if parameter_values.size != 1 or parameter_values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("canonicalization pullback requires one real parameter")
    mode = str(truncation_mode).lower()
    if mode not in {"exact", "approximate", "blocked", "non_differentiable"}:
        raise ValueError(
            "truncation_mode must be exact, approximate, blocked, or non_differentiable"
        )
    cutoff_value = float(cutoff)
    if not np.isfinite(cutoff_value) or cutoff_value < 0.0:
        raise ValueError("truncation cutoff must be a finite non-negative number")

    cpu_device = jax.devices("cpu")[0]
    theta = jax.device_put(
        jnp.asarray(float(parameter_values.reshape(-1)[0]), dtype=jnp.float64),
        cpu_device,
    )
    weights = jax.device_put(
        jnp.asarray(((1.0, -0.3), (0.2, 0.7)), dtype=jnp.float64),
        cpu_device,
    )

    def bond_matrix(value: Any) -> Any:
        return jnp.asarray(
            ((jnp.cos(value), 0.0), (0.0, 0.25 * jnp.sin(value))),
            dtype=jnp.float64,
        )

    def canonicalized_score(value: Any) -> Any:
        matrix = bond_matrix(value)
        q_factor, r_factor = jnp.linalg.qr(matrix)
        reconstructed = jnp.matmul(q_factor, r_factor, precision="highest")
        return jnp.sum(weights * reconstructed)

    def rank_one_truncated_score(value: Any) -> Any:
        matrix = bond_matrix(value)
        u_factor, singular, vh_factor = jnp.linalg.svd(matrix, full_matrices=False)
        reconstructed = jnp.matmul(
            u_factor[:, :1] * singular[:1],
            vh_factor[:1, :],
            precision="highest",
        )
        return jnp.sum(weights * reconstructed)

    matrix = bond_matrix(theta)
    q_factor, r_factor = jnp.linalg.qr(matrix)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    singular_values_host = np.asarray(jax.device_get(singular_values))
    discarded_weight = (
        float(singular_values_host[-1] ** 2) if mode == "approximate" else 0.0
    )
    canonical_value, canonical_pullback = jax.vjp(canonicalized_score, theta)
    canonical_gradient = canonical_pullback(
        jnp.asarray(1.0, dtype=canonical_value.dtype)
    )[0]
    truncation_gradient: Any | None = None
    if mode == "approximate":
        truncated_value, truncated_pullback = jax.vjp(rank_one_truncated_score, theta)
        truncation_gradient = truncated_pullback(
            jnp.asarray(1.0, dtype=truncated_value.dtype)
        )[0]
    elif mode == "exact":
        truncation_gradient = canonical_gradient

    pullback_executed = mode in {"exact", "approximate"}
    if mode == "exact":
        truncation_status = "exact"
        gradient_semantics = "exact_untruncated_svd_pullback"
        truncation_blockers: tuple[str, ...] = ()
    elif mode == "approximate":
        truncation_status = "approximate"
        gradient_semantics = "approximate_rank_one_truncated_pullback"
        truncation_blockers = (
            "mps_truncation_pullback_approximate_not_exact",
            "mps_truncation_discarded_weight_nonzero",
        )
    elif mode == "non_differentiable":
        truncation_status = "non_differentiable"
        gradient_semantics = "blocked_at_singular_value_selection_boundary"
        truncation_blockers = ("mps_truncation_selection_non_differentiable",)
    else:
        truncation_status = "blocked"
        gradient_semantics = "unsupported_truncation_pullback"
        truncation_blockers = ("mps_truncation_pullback_unsupported",)

    q_host = np.asarray(jax.device_get(q_factor))
    r_host = np.asarray(jax.device_get(r_factor))
    summary = _execute_minimal_mps_boundary_gate_adjoint_pullback(float(theta))
    blockers = (
        tuple(
            blocker
            for blocker in summary["blockers"]
            if blocker != "mps_general_bond_canonicalization_pullback_pending"
        )
        + truncation_blockers
    )
    canonicalization_evidence = {
        "status": "executed",
        "strategy": "boundary_bond_qr_pullback",
        "classification": "exact",
        "bond_id": 0,
        "left_owner_rank": 0,
        "right_owner_rank": 1,
        "q_shape": tuple(int(dim) for dim in q_host.shape),
        "r_shape": tuple(int(dim) for dim in r_host.shape),
        "dtype": str(q_host.dtype),
        "pullback_execution_status": "executed",
        "gradient_value": float(jax.device_get(canonical_gradient)),
        "full_local_mps_replay": False,
        "blockers": (),
    }
    truncation_evidence = {
        "status": truncation_status,
        "classification": truncation_status,
        "bond_id": 0,
        "left_owner_rank": 0,
        "right_owner_rank": 1,
        "cutoff": cutoff_value,
        "original_bond_dimension": 2,
        "retained_bond_dimension": 1 if mode == "approximate" else 2,
        "singular_values": tuple(float(item) for item in singular_values_host),
        "discarded_weight": discarded_weight,
        "gradient_semantics": gradient_semantics,
        "gradient_value": (
            float(jax.device_get(truncation_gradient))
            if truncation_gradient is not None
            else None
        ),
        "pullback_execution_status": "executed" if pullback_executed else "blocked",
        "exact_gradient_claim_allowed": mode == "exact",
        "blockers": truncation_blockers,
    }
    summary.update(
        {
            "executor": "minimal_mps_canonicalization_truncation_pullback",
            "canonicalization_backward_strategy": canonicalization_evidence,
            "truncation_gradient_metadata": truncation_evidence,
            "canonicalization_pullback_execution": "executed",
            "truncation_pullback_execution": (
                "executed" if pullback_executed else "blocked"
            ),
            "canonicalization_gradient": float(jax.device_get(canonical_gradient)),
            "full_mps_reconstruction_count": 0,
            "replicated_mps_autograd": False,
            "statevector_fallback": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": blockers,
            "gradient_blockers": blockers,
            "blockers": blockers,
        }
    )
    return _attach_mps_backward_readiness(summary)


def _execute_minimal_mps_sharded_optimizer_step(
    parameters: Any,
    *,
    learning_rate: float = 0.05,
    execution_backend: str = "auto",
) -> dict[str, Any]:
    """Execute one owner-rank SGD step for the constrained sharded MPS path."""

    import numpy as np

    parameter_values = np.asarray(parameters)
    if parameter_values.shape != (2,) or parameter_values.dtype.kind not in {
        "f",
        "i",
        "u",
    }:
        raise ValueError("minimal MPS sharded optimizer requires two real parameters")
    parameter_values = parameter_values.astype(np.float64, copy=False)
    learning_rate_value = float(learning_rate)
    if not np.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be a positive finite number")

    summary = _execute_minimal_mps_sharded_backward(
        parameter_values,
        execution_backend=execution_backend,
    )
    accelerator_backed = summary["claim_evidence_type"] == "production_runtime"
    optimizer_backend = "owner_local_cpu_sgd"
    if accelerator_backed:
        jax, jnp = _require_jax()
        devices = tuple(
            device
            for device in jax.local_devices()
            if str(getattr(device, "platform", "unknown")).lower() != "cpu"
        )[:2]
        jax.config.update("jax_enable_x64", True)

        def rank_optimizer_step(theta: Any) -> tuple[Any, Any]:
            local_z = jnp.cos(theta)
            remote_z = jax.lax.ppermute(
                local_z,
                "mps_rank",
                ((0, 1), (1, 0)),
            )
            gradient = -jnp.sin(theta) * remote_z
            return gradient, theta - learning_rate_value * gradient

        mapped_step = jax.pmap(
            rank_optimizer_step,
            axis_name="mps_rank",
            devices=devices,
        )
        gradients_array, updated_array = mapped_step(
            jnp.asarray(parameter_values, dtype=jnp.float64)
        )
        gradients_array.block_until_ready()
        updated_array.block_until_ready()
        gradients = np.asarray(jax.device_get(gradients_array), dtype=np.float64)
        updated_parameters = np.asarray(jax.device_get(updated_array), dtype=np.float64)
        optimizer_backend = "jax_pmap_owner_local_sgd"
        optimizer_execution_scope = "single_node_accelerator"
        optimizer_execution_status = "production_executed"
    else:
        gradients = np.asarray(summary["gradient"], dtype=np.float64)
        updated_parameters = parameter_values - learning_rate_value * gradients
        optimizer_execution_scope = "development_cpu"
        optimizer_execution_status = "development_executed"

    ownership = tuple(
        {
            "rank": rank,
            "owner_rank": rank,
            "parameter_id": f"theta_{rank}",
            "parameter_indices": (rank,),
            "parameter_owner_rank": rank,
            "gradient_owner_rank": rank,
            "optimizer_state_owner_rank": rank,
            "update_owner_rank": rank,
            "optimizer_state": {"step": 1},
            "gradient_value": float(gradients[rank]),
            "parameter_value_before": float(parameter_values[rank]),
            "parameter_value_after": float(updated_parameters[rank]),
            "update_value": float(updated_parameters[rank] - parameter_values[rank]),
            "writeback_route": "rank_local_parameter_owner_writeback",
            "execution_status": optimizer_execution_status,
            "non_owner_update_writes": (),
        }
        for rank in range(2)
    )
    blockers = tuple(
        blocker
        for blocker in summary["blockers"]
        if blocker != "mps_optimizer_update_ownership_pending"
    ) + (
        "mps_optimizer_limited_to_single_owner_local_sgd_step",
        *(("mps_optimizer_development_cpu_only",) if not accelerator_backed else ()),
    )
    summary.update(
        {
            "executor": "minimal_mps_sharded_optimizer_step",
            "optimizer": "sgd",
            "optimizer_backend": optimizer_backend,
            "optimizer_execution_scope": optimizer_execution_scope,
            "optimizer_execution_status": optimizer_execution_status,
            "learning_rate": learning_rate_value,
            "training_step_count": 1,
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_state_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "parameter_ownership": ownership,
            "gradient_ownership": ownership,
            "optimizer_state_ownership": ownership,
            "optimizer_update_ownership": ownership,
            "updated_parameters": tuple(float(item) for item in updated_parameters),
            "non_owner_update_writes": (),
            "full_mps_reconstruction_count": 0,
            "replicated_mps_autograd": False,
            "statevector_fallback": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": blockers,
            "gradient_blockers": blockers,
            "blockers": blockers,
        }
    )
    measured = dict(summary["mps_measured_runtime_evidence"])
    measured_rank_memory = []
    optimizer_scalar_bytes = int(np.asarray(learning_rate_value).nbytes)
    for record in measured["rank_memory"]:
        updated_record = dict(record)
        rank = int(updated_record["rank"])
        updated_record["optimizer_state_bytes"] = optimizer_scalar_bytes
        updated_record["optimizer_update_buffer_bytes"] = int(
            np.asarray(updated_parameters[rank]).nbytes
        )
        updated_record["measured_peak_live_tensor_bytes"] = sum(
            int(updated_record[field])
            for field in (
                "forward_tensor_bytes",
                "backward_adjoint_bytes",
                "boundary_buffer_bytes",
                "parameter_gradient_bytes",
                "optimizer_state_bytes",
                "optimizer_update_buffer_bytes",
            )
        )
        measured_rank_memory.append(updated_record)
    summary["mps_measured_runtime_evidence"] = (
        _summarize_minimal_mps_measured_runtime_evidence(
            measured_rank_memory,
            measured["communication_events"],
            execution_scope=optimizer_execution_scope,
        )
    )
    return _attach_mps_backward_readiness(summary)


def _jax_mps_boundary_protocol(
    *,
    left_wire: int,
    right_wire: int,
    left_rank: int,
    right_rank: int,
    local_world_size: int,
    tensor_bytes: int,
) -> dict[str, Any]:
    tier = _communication_tier(left_rank, right_rank, local_world_size=local_world_size)
    return {
        "left_wire": int(left_wire),
        "right_wire": int(right_wire),
        "left_rank": int(left_rank),
        "right_rank": int(right_rank),
        "owner_rank": int(left_rank),
        "tier": tier,
        "transport": "local_simulated_boundary_exchange",
        "stages": (
            "exchange_boundary_site_tensors",
            "apply_jax_two_site_update",
            "scatter_updated_boundary_site_tensors",
        ),
        "tensor_bytes": int(tensor_bytes),
        "estimated_transfer_bytes": int(tensor_bytes * 2),
        "point_to_point_messages": 2,
        "collective_messages": 0,
    }
