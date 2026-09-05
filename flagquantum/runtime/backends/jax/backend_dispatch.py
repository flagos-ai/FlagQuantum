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
from .mps_planning import _mps_plan
from .planning_core import JAXDistributedQuantumPlan, _as_ir
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)
from .runtime_environment import (
    _jax_available_local_devices,
    _require_jax,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
)
from .statevector_gradient_records import _statevector_plan
from .tensor_network_planning import _tensor_network_plan


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
