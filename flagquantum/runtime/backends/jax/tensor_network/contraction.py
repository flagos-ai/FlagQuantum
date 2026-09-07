"""Greedy tensor-network contraction, slice reduction, and Pauli helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .....simulation.jax.tensor_network import (
    _jax_einsum_by_labels,
    _jax_einsum_reorder,
    jax_slice_tensor_by_labels,
)
from ..common import product_int as _product
from ..runtime_environment import (
    _jax_available_local_devices,
    _require_jax,
)
from .records import JAXTensorNetworkNode


def _jax_tn_label_dims(nodes: Sequence[JAXTensorNetworkNode]) -> dict[int, int]:
    dims: dict[int, int] = {}
    for node in nodes:
        for label, dim in zip(node.labels, node.tensor.shape):
            label = int(label)
            dim = int(dim)
            if label in dims and dims[label] != dim:
                raise ValueError(
                    f"Inconsistent dimension for tensor-network label {label}."
                )
            dims[label] = dim
    return dims


def _jax_tn_label_counts(nodes: Sequence[JAXTensorNetworkNode]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        for label in node.labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    return counts


def _jax_tn_pair_output_labels(
    left: JAXTensorNetworkNode,
    right: JAXTensorNetworkNode,
    *,
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, ...]:
    labels = tuple(dict.fromkeys(left.labels + right.labels))
    return tuple(
        label
        for label in labels
        if not (
            label in left.labels
            and label in right.labels
            and label_counts.get(label, 0) == 2
            and label not in final_outputs
        )
    )


def _jax_tn_estimate_pair(
    left: JAXTensorNetworkNode,
    right: JAXTensorNetworkNode,
    *,
    dims: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    output_labels = _jax_tn_pair_output_labels(
        left,
        right,
        label_counts=label_counts,
        final_outputs=final_outputs,
    )
    union_labels = tuple(dict.fromkeys(left.labels + right.labels))
    cost = _product(dims[label] for label in union_labels)
    intermediate_size = _product(dims[label] for label in output_labels)
    contracted = tuple(label for label in union_labels if label not in output_labels)
    return cost, intermediate_size, output_labels, contracted


def _jax_tn_choose_greedy_pair(
    nodes: Sequence[JAXTensorNetworkNode],
    *,
    dims: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, int, tuple[int, ...]]:
    best_score: tuple[Any, ...] | None = None
    best: tuple[int, int, tuple[int, ...]] | None = None
    for left_idx in range(len(nodes)):
        for right_idx in range(left_idx + 1, len(nodes)):
            left = nodes[left_idx]
            right = nodes[right_idx]
            cost, intermediate, pair_outputs, contracted = _jax_tn_estimate_pair(
                left,
                right,
                dims=dims,
                label_counts=label_counts,
                final_outputs=final_outputs,
            )
            shared = len(set(left.labels) & set(right.labels))
            score = (
                -len(contracted),
                intermediate,
                cost,
                -shared,
                left_idx,
                right_idx,
                pair_outputs,
            )
            if best_score is None or score < best_score:
                best_score = score
                best = (left_idx, right_idx, pair_outputs)
    if best is None:
        raise ValueError("Tensor network requires at least two nodes to contract.")
    return best


def _jax_tn_einsum_pair_by_labels(
    left_tensor: Any,
    left_labels: Sequence[int],
    right_tensor: Any,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> Any:
    return _jax_einsum_by_labels(
        left_tensor,
        tuple(int(label) for label in left_labels),
        right_tensor,
        tuple(int(label) for label in right_labels),
        tuple(int(label) for label in output_labels),
        None,
    )


def _jax_tn_reorder_by_labels(
    tensor: Any,
    labels: Sequence[int],
    output_labels: Sequence[int],
) -> Any:
    return _jax_einsum_reorder(
        tensor,
        tuple(int(label) for label in labels),
        tuple(int(label) for label in output_labels),
        None,
    )


def _jax_tn_slice_nodes(
    nodes: Sequence[JAXTensorNetworkNode],
    assignments: Mapping[int, int],
) -> tuple[JAXTensorNetworkNode, ...]:
    sliced = []
    for node in nodes:
        tensor, labels = jax_slice_tensor_by_labels(
            node.tensor,
            node.labels,
            assignments,
        )
        sliced.append(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=labels,
                name=node.name,
                metadata=node.metadata,
            )
        )
    return tuple(sliced)


def _jax_zero_for_tn_output(
    nodes: Sequence[JAXTensorNetworkNode], output_labels: Sequence[int]
) -> Any:
    _, jnp = _require_jax()
    dims = _jax_tn_label_dims(nodes)
    reference = nodes[0].tensor if nodes else jnp.empty((), dtype=jnp.complex64)
    return jnp.zeros(
        tuple(dims[int(label)] for label in output_labels), dtype=reference.dtype
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
