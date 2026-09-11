"""Greedy local contraction of JAX tensor-network nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any

from .kernels import (
    _jax_einsum_by_labels,
    _jax_einsum_reorder,
    jax_slice_tensor_by_labels,
)
from .models import JAXTensorNetworkNode


def _label_dimensions(nodes: Sequence[JAXTensorNetworkNode]) -> dict[int, int]:
    dimensions: dict[int, int] = {}
    for node in nodes:
        for label, dimension in zip(node.labels, node.tensor.shape):
            label = int(label)
            dimension = int(dimension)
            if label in dimensions and dimensions[label] != dimension:
                raise ValueError(
                    f"Inconsistent dimension for tensor-network label {label}."
                )
            dimensions[label] = dimension
    return dimensions


def _label_counts(nodes: Sequence[JAXTensorNetworkNode]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        for label in node.labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    return counts


def _pair_output_labels(
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


def _estimate_pair(
    left: JAXTensorNetworkNode,
    right: JAXTensorNetworkNode,
    *,
    dimensions: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    output_labels = _pair_output_labels(
        left,
        right,
        label_counts=label_counts,
        final_outputs=final_outputs,
    )
    union_labels = tuple(dict.fromkeys(left.labels + right.labels))
    cost = prod(dimensions[label] for label in union_labels)
    intermediate_size = prod(dimensions[label] for label in output_labels)
    contracted = tuple(label for label in union_labels if label not in output_labels)
    return cost, intermediate_size, output_labels, contracted


def _choose_greedy_pair(
    nodes: Sequence[JAXTensorNetworkNode],
    *,
    dimensions: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, int, tuple[int, ...]]:
    best_score: tuple[Any, ...] | None = None
    best: tuple[int, int, tuple[int, ...]] | None = None
    for left_index in range(len(nodes)):
        for right_index in range(left_index + 1, len(nodes)):
            left = nodes[left_index]
            right = nodes[right_index]
            cost, intermediate, pair_outputs, contracted = _estimate_pair(
                left,
                right,
                dimensions=dimensions,
                label_counts=label_counts,
                final_outputs=final_outputs,
            )
            shared = len(set(left.labels) & set(right.labels))
            score = (
                -len(contracted),
                intermediate,
                cost,
                -shared,
                left_index,
                right_index,
                pair_outputs,
            )
            if best_score is None or score < best_score:
                best_score = score
                best = (left_index, right_index, pair_outputs)
    if best is None:
        raise ValueError("Tensor network requires at least two nodes to contract.")
    return best


def slice_nodes(
    nodes: Sequence[JAXTensorNetworkNode], assignments: Mapping[int, int]
) -> tuple[JAXTensorNetworkNode, ...]:
    """Apply fixed label assignments to a sequence of nodes."""

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


def zero_for_output(
    nodes: Sequence[JAXTensorNetworkNode], output_labels: Sequence[int]
) -> Any:
    """Create a zero value with the network's output shape and dtype."""

    import jax.numpy as jnp

    dimensions = _label_dimensions(nodes)
    reference = nodes[0].tensor if nodes else jnp.empty((), dtype=jnp.complex64)
    return jnp.zeros(
        tuple(dimensions[int(label)] for label in output_labels),
        dtype=reference.dtype,
    )


def contract_nodes_greedy(
    nodes: Sequence[JAXTensorNetworkNode], output_labels: Sequence[int]
) -> Any:
    """Contract nodes locally using a deterministic greedy pair order."""

    active = list(nodes)
    final_outputs = set(int(label) for label in output_labels)
    if not active:
        raise ValueError("Cannot contract an empty tensor network.")
    while len(active) > 1:
        dimensions = _label_dimensions(active)
        counts = _label_counts(active)
        left_index, right_index, pair_outputs = _choose_greedy_pair(
            active,
            dimensions=dimensions,
            label_counts=counts,
            final_outputs=final_outputs,
        )
        left = active[left_index]
        right = active[right_index]
        tensor = _jax_einsum_by_labels(
            left.tensor,
            left.labels,
            right.tensor,
            right.labels,
            pair_outputs,
            None,
        )
        new_node = JAXTensorNetworkNode(
            tensor=tensor,
            labels=pair_outputs,
            name=f"({left.name},{right.name})",
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(new_node)
    final = active[0]
    if final.labels != tuple(output_labels):
        return _jax_einsum_reorder(
            final.tensor,
            final.labels,
            tuple(int(label) for label in output_labels),
            None,
        )
    return final.tensor


def contract_assigned_slices(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
) -> Any:
    """Contract and sum the slices assigned to one local worker."""

    partial = zero_for_output(nodes, output_labels)
    for _, assignments in tasks:
        partial = partial + contract_nodes_greedy(
            slice_nodes(nodes, dict(assignments)), output_labels
        )
    return partial
