# ruff: noqa: F401, F821
"""Tensor-network slice records, contraction kernels, and forward execution."""

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


@dataclass
class JAXTNSliceRankState:
    """Rank-local tensor-network slice tasks and partial contraction result."""

    rank: int
    tasks: tuple[Mapping[str, Any], ...]
    partial: Any

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "slice_task_count": len(self.tasks),
            "slice_tasks": self.tasks,
            "partial_shape": tuple(int(dim) for dim in self.partial.shape),
            "partial_bytes": _jax_array_nbytes(self.partial),
            "dtype": str(self.partial.dtype),
        }


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
    _, jnp = _require_jax()
    labels = tuple(
        dict.fromkeys(tuple(left_labels) + tuple(right_labels) + tuple(output_labels))
    )
    if len(labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "Pair contraction rank exceeds local JAX einsum label capacity."
        )
    mapping = {label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    equation = (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, left_tensor, right_tensor)


def _jax_tn_reorder_by_labels(
    tensor: Any,
    labels: Sequence[int],
    output_labels: Sequence[int],
) -> Any:
    _, jnp = _require_jax()
    all_labels = tuple(dict.fromkeys(tuple(labels) + tuple(output_labels)))
    if len(all_labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "Final contraction rank exceeds local JAX einsum label capacity."
        )
    mapping = {
        label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(all_labels)
    }
    equation = (
        "".join(mapping[label] for label in labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, tensor)


def _jax_tn_slice_nodes(
    nodes: Sequence[JAXTensorNetworkNode],
    assignments: Mapping[int, int],
) -> tuple[JAXTensorNetworkNode, ...]:
    _, jnp = _require_jax()
    sliced = []
    for node in nodes:
        tensor = node.tensor
        labels = list(node.labels)
        for label, value in assignments.items():
            if int(label) not in labels:
                continue
            axis = labels.index(int(label))
            tensor = jnp.take(tensor, int(value), axis=axis)
            labels.pop(axis)
        sliced.append(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=tuple(labels),
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


def _jax_tn_task_summary(
    task: tuple[int, tuple[tuple[int, int], ...]], task_index: int
) -> dict[str, Any]:
    rank, assignments = task
    return {
        "task_index": int(task_index),
        "rank": int(rank),
        "assignments": tuple((int(label), int(value)) for label, value in assignments),
    }


def _jax_reduced_tn_output_to_torch_state(
    output: Any,
    *,
    n_wires: int,
    bsz: int,
    complex_bytes: int,
) -> Any:
    import numpy as np

    torch = _require_torch()
    dtype = _torch_complex_dtype(complex_bytes)
    state = torch.as_tensor(
        np.asarray(output).copy(), dtype=dtype, device=torch.device("cpu")
    )
    return state.reshape(int(bsz), 2 ** int(n_wires))


def _resolve_tn_compute_backend(
    compute_backend: str, policy: DistributedBackendPolicy
) -> str:
    backend = str(compute_backend).lower()
    if backend in {"auto", ""}:
        if policy.profile == "production" and policy.jax_backend in {
            "pmap",
            "jax_pmap",
        }:
            return "pmap"
        if policy.profile == "production" and policy.jax_backend in {
            "shard_map",
            "jax_shard_map",
        }:
            return "shard_map"
        return "local_simulated"
    if backend in {"local", "local_simulated", "local_simulated_slice_compute"}:
        return "local_simulated"
    if backend in {"pmap", "jax_pmap", "jax_pmap_slice_contraction"}:
        return "pmap"
    if backend in {"shard_map", "jax_shard_map"}:
        return "shard_map"
    raise ValueError(
        "compute_backend must be 'auto', 'local_simulated', 'pmap', or 'shard_map'."
    )


def run_jax_sharded_tensor_network(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    compute_backend: str = "auto",
    collective_backend: str = "auto",
) -> JAXShardedTensorNetworkResult:
    """Execute JAX tensor-network contraction with slice tasks sharded by rank."""

    from ....simulation.tensor import build_tensor_network

    torch = _require_torch()
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_collective_backend = _resolve_collective_backend(
        collective_backend, policy
    )
    resolved_compute_backend = _resolve_tn_compute_backend(compute_backend, policy)
    if (
        resolved_compute_backend == "pmap"
        and resolved_collective_backend == "local_simulated"
    ):
        resolved_collective_backend = "pmap"
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    torch_dtype = _torch_complex_dtype(complex_bytes)
    jax_dtype = _jax_complex_dtype(complex_bytes)
    jax_device = _resolve_jax_device(device)
    plan = build_tensor_network(
        circuit_or_ir,
        bsz=bsz,
        device="cpu",
        dtype=torch_dtype,
    )
    slicing = plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, resolved_world_size)
    active_ranks = tuple(sorted({int(rank) for rank, _ in tasks}))
    if resolved_world_size > 1 and (int(slicing.n_slices) < 2 or len(active_ranks) < 2):
        raise RuntimeError(
            "JAX sharded tensor-network execution requires at least two slice tasks assigned to multiple ranks. "
            "Pass sliced_labels or max_intermediate_size so the contraction is actually partitioned; "
            "FlagQuantum will not label an unsliced full contraction as distributed."
        )
    jax_nodes = _jax_nodes_from_torch_nodes(
        plan.nodes, dtype=jax_dtype, device=jax_device
    )
    task_summaries = tuple(
        _jax_tn_task_summary(task, index) for index, task in enumerate(tasks)
    )
    partial_values, reduced, compute_execution, collective_execution = (
        _jax_contract_tensor_slices_by_backend(
            jax_nodes,
            plan.output_labels,
            tasks,
            world_size=resolved_world_size,
            compute_backend=resolved_compute_backend,
            collective_backend=resolved_collective_backend,
        )
    )
    rank_partials = []
    for rank, partial in enumerate(partial_values):
        rank_summary_by_assignment = tuple(
            task_summaries[index]
            for index, task in enumerate(tasks)
            if int(task[0]) == rank
        )
        rank_partials.append(
            JAXTNSliceRankState(
                rank=rank, tasks=rank_summary_by_assignment, partial=partial
            )
        )
    reduced = _jnp_device_put(reduced, jax_device)
    jax_plan = plan_jax_distributed_quantum_backend(
        circuit_or_ir,
        mode="tensor_network",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        distributed_backend_policy=policy,
    )
    return JAXShardedTensorNetworkResult(
        rank_partials=tuple(rank_partials),
        reduced_output=reduced,
        slicing=slicing,
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=plan.n_wires,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        local_world_size=resolved_local_world_size,
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        compute_backend=resolved_compute_backend,
        compute_execution=compute_execution,
        collective_backend=resolved_collective_backend,
        collective_execution=collective_execution,
    )


def _jax_tn_loss_from_output(
    output: Any,
    *,
    n_wires: int,
    bsz: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    _, jnp = _require_jax()
    state = output.reshape(int(bsz), 2 ** int(n_wires))
    if str(observable) == "state_norm":
        return jnp.real(jnp.sum(jnp.conj(state) * state))
    if str(observable) != "z_sum":
        raise ValueError(
            "JAX sliced TN reverse mode currently supports observable='z_sum' or 'state_norm'."
        )
    wires = (
        tuple(range(int(n_wires)))
        if observable_wires is None
        else tuple(int(wire) for wire in observable_wires)
    )
    probs = jnp.abs(state) ** 2
    indices = jnp.arange(2 ** int(n_wires))
    total = jnp.zeros((), dtype=probs.real.dtype)
    for wire in wires:
        bits = (indices >> (int(n_wires) - 1 - int(wire))) & 1
        weights = 1.0 - 2.0 * bits.astype(probs.real.dtype)
        total = total + jnp.sum(probs * weights.reshape(1, -1))
    return total
