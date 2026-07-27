# ruff: noqa: F401, F821
"""Greedy tensor-network contraction, slice reduction, and Pauli helpers."""

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


def _jax_contract_nodes_greedy(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
) -> Any:
    active = list(nodes)
    final_outputs = set(int(label) for label in output_labels)
    if not active:
        raise ValueError("Cannot contract an empty tensor network.")
    while len(active) > 1:
        dims = _jax_tn_label_dims(active)
        counts = _jax_tn_label_counts(active)
        left_idx, right_idx, pair_outputs = _jax_tn_choose_greedy_pair(
            active,
            dims=dims,
            label_counts=counts,
            final_outputs=final_outputs,
        )
        left = active[left_idx]
        right = active[right_idx]
        tensor = _jax_tn_einsum_pair_by_labels(
            left.tensor, left.labels, right.tensor, right.labels, pair_outputs
        )
        new_node = JAXTensorNetworkNode(
            tensor=tensor,
            labels=pair_outputs,
            name=f"({left.name},{right.name})",
        )
        for index in sorted((left_idx, right_idx), reverse=True):
            active.pop(index)
        active.append(new_node)
    final = active[0]
    if final.labels != tuple(output_labels):
        return _jax_tn_reorder_by_labels(final.tensor, final.labels, output_labels)
    return final.tensor


def _jax_contract_assigned_tensor_slices(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
) -> Any:
    partial = _jax_zero_for_tn_output(nodes, output_labels)
    for _, assignments in tasks:
        subnodes = _jax_tn_slice_nodes(nodes, dict(assignments))
        partial = partial + _jax_contract_nodes_greedy(subnodes, output_labels)
    return partial


def _tasks_by_rank_from_slicing(slicing: Any, world_size: int) -> tuple[int, ...]:
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, int(world_size))
    return tuple(
        sum(1 for rank, _ in tasks if int(rank) == target)
        for target in range(int(world_size))
    )


def _jax_reduce_rank_partials(
    partials: Sequence[Any],
    *,
    collective_backend: str,
) -> tuple[Any, str]:
    jax, jnp = _require_jax()
    partials = tuple(partials)
    if not partials:
        raise ValueError("Cannot reduce an empty partial list.")
    backend = str(collective_backend)
    if backend == "local_simulated":
        return jnp.sum(jnp.stack(partials, axis=0), axis=0), "local_simulated_psum"
    if backend == "shard_map":
        # Keep a strict failure until the production shard_map axis resources
        # are plumbed through; silently using pmap would overstate coverage.
        raise RuntimeError(
            "JAX shard_map tensor-network reduction requires production mesh axis resources; "
            "use collective_backend='pmap' for device-local psum or 'local_simulated' for CPU development."
        )
    devices = _jax_available_local_devices()
    if len(devices) < len(partials):
        raise RuntimeError(
            f"JAX pmap tensor-network reduction requires at least {len(partials)} local JAX devices, "
            f"but only {len(devices)} are visible. This path fails closed instead of simulating production collectives."
        )
    sharded = jax.device_put_sharded(list(partials), list(devices[: len(partials)]))

    def _psum(local_value: Any) -> Any:
        return jax.lax.psum(local_value, axis_name="fq_rank")

    reduced = jax.pmap(_psum, axis_name="fq_rank")(sharded)
    return reduced[0], "jax_pmap_psum"


def _jax_contract_tensor_slices_with_pmap(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
    *,
    world_size: int,
) -> tuple[tuple[Any, ...], Any, str, str]:
    jax, jnp = _require_jax()
    world_size = max(1, int(world_size))
    devices = _jax_available_local_devices()
    if len(devices) < world_size:
        raise RuntimeError(
            f"JAX pmap tensor-network slice compute requires at least {world_size} local JAX devices, "
            f"but only {len(devices)} are visible. This path fails closed instead of running replicated work."
        )
    tasks_by_rank = tuple(
        tuple(task for task in tasks if int(task[0]) == rank)
        for rank in range(world_size)
    )

    branches = []
    for rank_tasks in tasks_by_rank:
        static_rank_tasks = tuple(rank_tasks)

        def _branch(
            _operand: Any,
            rank_tasks: tuple[
                tuple[int, tuple[tuple[int, int], ...]], ...
            ] = static_rank_tasks,
        ) -> Any:
            return _jax_contract_assigned_tensor_slices(
                nodes, output_labels, rank_tasks
            )

        branches.append(_branch)

    operand = _jax_zero_for_tn_output(nodes, output_labels)

    def _compute(rank_index: Any) -> tuple[Any, Any]:
        local_partial = jax.lax.switch(rank_index, tuple(branches), operand)
        reduced = jax.lax.psum(local_partial, axis_name="fq_rank")
        return local_partial, reduced

    rank_indices = [jnp.asarray(rank, dtype=jnp.int32) for rank in range(world_size)]
    sharded_indices = jax.device_put_sharded(rank_indices, list(devices[:world_size]))
    local_partials, reduced_per_rank = jax.pmap(_compute, axis_name="fq_rank")(
        sharded_indices
    )
    return (
        tuple(local_partials),
        reduced_per_rank[0],
        "jax_pmap_slice_contraction",
        "jax_pmap_psum",
    )


def _jax_parameter_array_from_input(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    jax, jnp = _require_jax()
    torch = _require_torch()
    real_dtype = _jax_real_dtype(complex_bytes)
    if torch.is_tensor(parameters):
        tensor = parameters.detach().contiguous()
        try:
            import jax.dlpack

            return jnp.asarray(jax.dlpack.from_dlpack(tensor), dtype=real_dtype)
        except Exception:
            return jnp.asarray(np.asarray(tensor.cpu()).copy(), dtype=real_dtype)
    del jax
    return jnp.asarray(parameters, dtype=real_dtype)


def _torch_parameters_for_static_build(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    torch = _require_torch()
    if torch.is_tensor(parameters):
        return parameters.detach().cpu()
    dtype = torch.float64 if int(complex_bytes) == 16 else torch.float32
    return torch.as_tensor(
        np.asarray(parameters).copy(), dtype=dtype, device=torch.device("cpu")
    )


def _pauli_ops_from_term(term_ops: Sequence[tuple[int, str]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for wire, name in term_ops:
        normalized = str(name).lower()
        if normalized != "i":
            out[int(wire)] = normalized
    return out
