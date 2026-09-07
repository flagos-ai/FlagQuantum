"""Owner-rank parameter VJP and boundary-gate adjoint pullbacks."""

from __future__ import annotations

from typing import Any, Sequence

from .....simulation.jax.mps.pullbacks import (
    jax_mps_boundary_rxx_pullback,
    jax_mps_owner_local_vjp,
)
from ..mps_backward import _execute_minimal_mps_sharded_backward
from ..release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from ..runtime_environment import _require_jax
from .boundary_exchange import _summarize_local_mps_boundary_adjoint_exchange


def _execute_minimal_mps_owner_rank_parameter_vjp(
    parameters: Any,
    *,
    gate_kinds: Sequence[str] = ("one_site_ry", "same_shard_two_site_rxx"),
) -> dict[str, Any]:
    """Execute constrained owner-rank MPS VJPs without global replay."""

    import numpy as np

    jax, jnp = _require_jax()
    jax.config.update("jax_enable_x64", True)
    parameter_values = np.asarray(parameters)
    if parameter_values.shape != (2,):
        raise ValueError(
            "owner-rank MPS VJP requires exactly two rank-owned parameters"
        )
    if parameter_values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("owner-rank MPS VJP requires real parameters")
    parameter_values = parameter_values.astype(np.float64, copy=False)
    normalized_gate_kinds = tuple(str(item).lower() for item in gate_kinds)
    supported_gate_kinds = {"one_site_ry", "same_shard_two_site_rxx"}
    if len(normalized_gate_kinds) != 2:
        raise ValueError("owner-rank MPS VJP requires one gate kind per rank")
    if any(item not in supported_gate_kinds for item in normalized_gate_kinds):
        raise ValueError(
            "owner-rank MPS VJP supports only one_site_ry and "
            "same_shard_two_site_rxx; boundary-crossing gates are unsupported"
        )

    local_scores = np.cos(parameter_values)
    received_cotangents = local_scores[::-1].copy()
    gradients = np.empty_like(parameter_values)
    local_tensors: list[dict[str, Any]] = []
    ownership: list[dict[str, Any]] = []
    for rank, (theta, gate_kind) in enumerate(
        zip(parameter_values, normalized_gate_kinds)
    ):
        theta_value = jnp.asarray(theta)
        gradient = jax_mps_owner_local_vjp(
            theta_value,
            received_cotangents[rank],
            gate_kind=gate_kind,
        )
        gradients[rank] = float(jax.device_get(gradient))
        half_angle = theta / 2.0
        if gate_kind == "one_site_ry":
            tensor = np.asarray(
                (np.cos(half_angle), np.sin(half_angle)),
                dtype=np.complex128,
            ).reshape((1, 2, 1))
            gate_name = "ry"
            wires = (2 * rank,)
        else:
            tensor = np.asarray(
                (np.cos(half_angle), 0.0, 0.0, -1j * np.sin(half_angle)),
                dtype=np.complex128,
            ).reshape((1, 2, 2, 1))
            gate_name = "rxx"
            wires = (2 * rank, 2 * rank + 1)
        gate_id = f"rank_{rank}_{gate_name}_0"
        parameter_id = f"theta_{rank}"
        local_tensors.append(
            {
                "rank": rank,
                "gate_id": gate_id,
                "gate_kind": gate_kind,
                "wires": wires,
                "shape": tuple(int(item) for item in tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": int(tensor.nbytes),
            }
        )
        ownership.append(
            {
                "rank": rank,
                "owner_rank": rank,
                "parameter_id": parameter_id,
                "gate_id": gate_id,
                "gate_kind": gate_kind,
                "parameter_indices": (rank,),
                "parameter_owner_rank": rank,
                "gradient_owner_rank": rank,
                "site_range": (2 * rank, 2 * rank + 1),
                "wires": wires,
                "local_tensor_shape": tuple(int(item) for item in tensor.shape),
                "local_tensor_dtype": str(tensor.dtype),
                "gradient_value": float(gradients[rank]),
                "gradient_bytes": int(gradients[rank].nbytes),
                "ownership_semantics": "rank_owned_gradient",
                "gradient_route": "owner_rank_local_tensor_vjp",
                "vjp_execution_status": "executed",
                "non_owner_gradient_writes": (),
                "replicated_autograd": False,
                "full_local_mps_replay": False,
                "statevector_fallback": False,
                "blockers": (),
            }
        )

    summary = _execute_minimal_mps_sharded_backward(
        parameter_values,
        execution_backend="cpu",
    )
    blockers = tuple(
        blocker
        for blocker in summary["blockers"]
        if blocker != "mps_minimal_backward_gate_set_limited_to_rank_local_ry"
    ) + (
        "mps_owner_rank_vjp_gate_set_limited_to_one_site_ry_and_same_shard_rxx",
        "mps_boundary_crossing_two_site_parameter_vjp_pending",
        "mps_owner_rank_vjp_development_cpu_only",
    )
    summary.update(
        {
            "executor": "minimal_mps_owner_rank_parameter_vjp",
            "execution_classification": "cpu_owner_rank_vjp_development_execution",
            "owner_rank_vjp_execution": "executed",
            "supported_gate_set": ("ry", "rxx_same_shard"),
            "gate_kinds": normalized_gate_kinds,
            "value": float(np.prod(local_scores)),
            "gradient": tuple(float(item) for item in gradients),
            "local_vjp_tensors": tuple(local_tensors),
            "parameter_gradient_ownership": tuple(ownership),
            "parameter_ownership": tuple(ownership),
            "gradient_ownership": tuple(ownership),
            "non_owner_gradient_writes": (),
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


def _execute_minimal_mps_boundary_gate_adjoint_pullback(
    parameter: Any,
    *,
    left_owner_rank: int = 0,
    right_owner_rank: int = 1,
    communication_route: str = "local_cpu_point_to_point_copy",
) -> dict[str, Any]:
    """Execute a two-rank boundary ``RXX`` pullback without full-MPS replay."""

    import numpy as np

    jax, jnp = _require_jax()
    jax.config.update("jax_enable_x64", True)
    parameter_values = np.asarray(parameter)
    if parameter_values.size != 1 or parameter_values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("boundary MPS gate pullback requires one real RXX parameter")
    if (int(left_owner_rank), int(right_owner_rank)) != (0, 1):
        raise ValueError(
            "boundary MPS gate pullback requires adjacent owner ranks (0, 1)"
        )
    if not str(communication_route).strip():
        raise ValueError("boundary MPS gate pullback requires a communication route")

    cpu_device = jax.devices("cpu")[0]
    theta = jax.device_put(
        jnp.asarray(float(parameter_values.reshape(-1)[0]), dtype=jnp.float64),
        cpu_device,
    )
    complex_dtype = jnp.complex128
    left_site = jax.device_put(
        jnp.asarray([1.0, 0.0], dtype=complex_dtype).reshape(1, 2, 1),
        cpu_device,
    )
    right_site = jax.device_put(
        jnp.asarray([1.0, 0.0], dtype=complex_dtype).reshape(1, 2, 1),
        cpu_device,
    )

    value, parameter_gradient, left_adjoint, right_adjoint = (
        jax_mps_boundary_rxx_pullback(theta, left_site, right_site)
    )
    value_host = float(jax.device_get(value))
    gradient_host = float(jax.device_get(parameter_gradient))
    left_adjoint_host = np.asarray(jax.device_get(left_adjoint))
    right_adjoint_host = np.asarray(jax.device_get(right_adjoint))
    expected_value = float(np.cos(float(theta)))
    expected_gradient = float(-np.sin(float(theta)))
    if not np.allclose(
        (value_host, gradient_host), (expected_value, expected_gradient), atol=1e-10
    ):
        raise RuntimeError(
            "boundary MPS gate pullback did not match the analytic local reference"
        )

    boundary_id = "edge_0:0-1"
    bond_id = 0
    records = (
        {
            "boundary_edge_id": boundary_id,
            "bond_id": bond_id,
            "direction": "left_to_right",
            "source_rank": 0,
            "target_rank": 1,
            "owned_site_ranges": {"source": (0,), "target": (1,)},
            "boundary_tensor_shape": tuple(
                int(dim) for dim in right_adjoint_host.shape
            ),
            "boundary_tensor_dtype": str(right_adjoint_host.dtype),
            "communication_bytes": int(right_adjoint_host.nbytes),
            "local_memory_bytes": int(right_adjoint_host.nbytes),
            "communication_primitive": str(communication_route),
            "boundary_gradient_ownership": {
                "owner_rank": 1,
                "ownership_semantics": "rank_owned_boundary_site_adjoint",
            },
            "boundary_adjoint_exchange_status": "executed",
            "pullback_execution_status": "executed",
            "blockers": (),
        },
        {
            "boundary_edge_id": boundary_id,
            "bond_id": bond_id,
            "direction": "right_to_left",
            "source_rank": 1,
            "target_rank": 0,
            "owned_site_ranges": {"source": (1,), "target": (0,)},
            "boundary_tensor_shape": tuple(int(dim) for dim in left_adjoint_host.shape),
            "boundary_tensor_dtype": str(left_adjoint_host.dtype),
            "communication_bytes": int(left_adjoint_host.nbytes),
            "local_memory_bytes": int(left_adjoint_host.nbytes),
            "communication_primitive": str(communication_route),
            "boundary_gradient_ownership": {
                "owner_rank": 0,
                "ownership_semantics": "rank_owned_boundary_site_adjoint",
            },
            "boundary_adjoint_exchange_status": "executed",
            "pullback_execution_status": "executed",
            "blockers": (),
        },
    )
    exchange = _summarize_local_mps_boundary_adjoint_exchange(
        records,
        world_size=2,
        expected_boundary_edges=1,
        execution_blockers=(),
    )
    base = _execute_minimal_mps_sharded_backward(
        [float(theta), 0.0], execution_backend="cpu"
    )
    blockers = tuple(
        blocker
        for blocker in base["blockers"]
        if blocker != "mps_minimal_backward_gate_set_limited_to_rank_local_ry"
    ) + (
        "mps_boundary_gate_pullback_single_rxx_only",
        "mps_boundary_gate_pullback_development_cpu_only",
        "mps_general_bond_canonicalization_pullback_pending",
    )
    parameter_record = {
        "rank": 0,
        "owner_rank": 0,
        "parameter_id": "theta_boundary_0",
        "gate_id": "boundary_rxx_0",
        "gate_kind": "boundary_two_site_rxx",
        "parameter_indices": (0,),
        "parameter_owner_rank": 0,
        "gradient_owner_rank": 0,
        "site_range": (0, 1),
        "wires": (0, 1),
        "touched_ranks": (0, 1),
        "gradient_value": gradient_host,
        "gradient_bytes": int(np.asarray(gradient_host).nbytes),
        "ownership_semantics": "rank_owned_gradient",
        "gradient_route": "boundary_adjoint_to_parameter_owner_vjp",
        "vjp_execution_status": "executed",
        "non_owner_gradient_writes": (),
        "blockers": (),
    }
    base.update(
        {
            "executor": "minimal_mps_boundary_gate_adjoint_pullback",
            "execution_classification": "cpu_boundary_pullback_development_execution",
            "boundary_gate_pullback_execution": "executed",
            "supported_gate_set": ("rxx_boundary",),
            "value": value_host,
            "gradient": (gradient_host,),
            "boundary_adjoint_exchange_evidence": exchange,
            "boundary_pullback_records": records,
            "boundary_gradient_ownership": (
                {"boundary_edge_id": boundary_id, "rank": 0, "role": "parameter_owner"},
                {
                    "boundary_edge_id": boundary_id,
                    "rank": 1,
                    "role": "remote_site_adjoint_owner",
                },
            ),
            "parameter_gradient_ownership": (parameter_record,),
            "parameter_ownership": (parameter_record,),
            "gradient_ownership": (parameter_record,),
            "non_owner_gradient_writes": (),
            "full_mps_reconstruction_count": 0,
            "replicated_mps_autograd": False,
            "statevector_fallback": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": blockers,
            "gradient_blockers": blockers,
            "blockers": blockers,
        }
    )
    return _attach_mps_backward_readiness(base)
