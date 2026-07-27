# ruff: noqa: F401, F821
"""Collective backend dispatch and representation-level planning entrypoint."""

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


def _jax_available_local_devices() -> tuple[Any, ...]:
    jax, _ = _require_jax()
    return tuple(jax.local_devices())


def _jax_contract_tensor_slices_by_backend(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
    *,
    world_size: int,
    compute_backend: str,
    collective_backend: str,
) -> tuple[tuple[Any, ...], Any, str, str]:
    compute_backend = str(compute_backend)
    if compute_backend == "shard_map":
        raise RuntimeError(
            "JAX shard_map tensor-network slice compute requires a production mesh and named axis resources; "
            "use compute_backend='pmap' for device-local sliced compute or 'local_simulated' for CPU development."
        )
    if compute_backend == "pmap":
        if collective_backend not in {"auto", "pmap", "local_simulated"}:
            raise ValueError(
                "compute_backend='pmap' supports collective_backend='auto', 'pmap', or 'local_simulated'."
            )
        return _jax_contract_tensor_slices_with_pmap(
            nodes,
            output_labels,
            tasks,
            world_size=world_size,
        )
    partials = tuple(
        _jax_contract_assigned_tensor_slices(
            nodes,
            output_labels,
            tuple(task for task in tasks if int(task[0]) == rank),
        )
        for rank in range(max(1, int(world_size)))
    )
    reduced, collective_execution = _jax_reduce_rank_partials(
        partials, collective_backend=collective_backend
    )
    return partials, reduced, "local_simulated_slice_compute", collective_execution


def plan_jax_distributed_quantum_backend(
    circuit_or_ir: Any,
    *,
    mode: str = "statevector",
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int = 8,
    max_bond: int | None = None,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
) -> JAXDistributedQuantumPlan:
    """Plan JAX distributed integration for a FlagQuantum circuit/IR.

    The returned plan is intentionally not a scalability claim.  It records how
    a future JAX pmap/shard_map executor must shard the workload and what is
    still missing before FlagQuantum can call the JAX path distributed capacity
    scaling.
    """

    normalized_mode = (
        "tensor_network" if mode in {"tn", "tensor_network"} else str(mode)
    )
    if normalized_mode not in {"statevector", "mps", "tensor_network"}:
        raise ValueError("mode must be 'statevector', 'mps', or 'tensor_network'.")
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    ir = _as_ir(circuit_or_ir)
    if normalized_mode == "statevector":
        return _statevector_plan(
            ir,
            policy=policy,
            world_size=resolved_world_size,
            local_world_size=resolved_local_world_size,
            bsz=bsz,
            complex_bytes=complex_bytes,
        )
    if normalized_mode == "mps":
        return _mps_plan(
            ir,
            policy=policy,
            world_size=resolved_world_size,
            local_world_size=resolved_local_world_size,
            bsz=bsz,
            complex_bytes=complex_bytes,
            max_bond=max_bond,
        )
    return _tensor_network_plan(
        ir,
        policy=policy,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
    )
