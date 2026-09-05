"""Dependency-light local JAX tensor-network contraction."""

from __future__ import annotations

from typing import Any

_JAX_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def jax_contract_nodes_greedy(
    nodes: list[tuple[Any, tuple[int, ...]]],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    active = list(nodes)
    final_outputs = set(output_labels)
    while len(active) > 1:
        counts: dict[int, int] = {}
        for _tensor, labels in active:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        pair = None
        for left_index in range(len(active)):
            for right_index in range(left_index + 1, len(active)):
                shared = set(active[left_index][1]) & set(active[right_index][1])
                if shared:
                    pair = (left_index, right_index)
                    break
            if pair is not None:
                break
        if pair is None:
            pair = (0, 1)
        left_index, right_index = pair
        left_tensor, left_labels = active[left_index]
        right_tensor, right_labels = active[right_index]
        pair_counts = dict(counts)
        for label in left_labels + right_labels:
            pair_counts[label] -= 1
        pair_outputs = tuple(
            label
            for label in tuple(dict.fromkeys(left_labels + right_labels))
            if label in final_outputs or pair_counts.get(label, 0) > 0
        )
        tensor = _jax_einsum_by_labels(
            left_tensor,
            left_labels,
            right_tensor,
            right_labels,
            pair_outputs,
            matmul_precision,
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append((tensor, pair_outputs))
    final_tensor, final_labels = active[0]
    if final_labels != tuple(output_labels):
        final_tensor = _jax_einsum_reorder(
            final_tensor, final_labels, output_labels, matmul_precision
        )
    return final_tensor


def _jax_einsum_by_labels(
    left_tensor: Any,
    left_labels: tuple[int, ...],
    right_tensor: Any,
    right_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    labels = tuple(dict.fromkeys(left_labels + right_labels + output_labels))
    if len(labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network pair contraction exceeded local einsum label capacity."
        )
    mapping = {label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    equation = (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, left_tensor, right_tensor, precision=matmul_precision)


def _jax_einsum_reorder(
    tensor: Any,
    labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    all_labels = tuple(dict.fromkeys(labels + output_labels))
    if len(all_labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network final contraction exceeded local einsum label capacity."
        )
    mapping = {
        label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(all_labels)
    }
    equation = (
        "".join(mapping[label] for label in labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, tensor, precision=matmul_precision)
