"""Distributed JAX tensor-network contraction orchestration."""

from __future__ import annotations

from typing import Any, Sequence

from .....simulation.jax.tensor_network.contraction import (
    contract_assigned_slices as _jax_contract_assigned_tensor_slices,
)
from .....simulation.jax.tensor_network.contraction import (
    zero_for_output as _jax_zero_for_tn_output,
)
from .....simulation.jax.tensor_network.models import JAXTensorNetworkNode
from ..runtime_environment import (
    _jax_available_local_devices,
    _require_jax,
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


def _pauli_ops_from_term(term_ops: Sequence[tuple[int, str]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for wire, name in term_ops:
        normalized = str(name).lower()
        if normalized != "i":
            out[int(wire)] = normalized
    return out
