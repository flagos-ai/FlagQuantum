# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Traversal follows quimb 1.15.0 TensorNetwork.antidiag_gauge/diagonal_reduce.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/tensor_core.py

"""Output-protected antidiagonal gauge and diagonal reduction on native nodes."""

from collections.abc import Sequence
from dataclasses import replace

import torch

from .models import TensorNetworkNode
from .simplification_axes import _simplification_axes
from .stages import einsum_reorder_by_labels
from .tensor_structure import find_tensor_diagonal


def _flip_edge(
    nodes: tuple[TensorNetworkNode, ...], label: int
) -> tuple[TensorNetworkNode, ...]:
    result = []
    for node in nodes:
        axes = tuple(axis for axis, value in enumerate(node.labels) if value == label)
        result.append(replace(node, tensor=node.tensor.flip(axes)) if axes else node)
    return tuple(result)


def _merge_edge(
    nodes: tuple[TensorNetworkNode, ...], removed: int, retained: int
) -> tuple[TensorNetworkNode, ...]:
    return tuple(
        (
            replace(
                node,
                labels=tuple(
                    retained if value == removed else value for value in node.labels
                ),
            )
            if removed in node.labels
            else node
        )
        for node in nodes
    )


def gauge_tensor_antidiagonals(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int] | None = None,
    *,
    atol: float = 1e-12,
    _cache: dict[tuple[int, int], torch.Tensor] | None = None,
    _identities: Sequence[int] | None = None,
) -> tuple[TensorNetworkNode, ...]:
    """Flip each selected internal edge at most once in source queue order.

    First-candidate/output protection follows the reference A pass. A global
    reversal of every occurrence of an internal edge is an exact gauge even
    when its detection uses atol. It preserves full contracted values and
    ambient gradients; no data is discarded. Unlike the source's first-axis
    Tensor.flip, repeated axes of the same edge are all flipped, so the gauge
    also preserves traces and repeated-label hyperedges. This difference is
    tested separately rather than presented as per-leaf source equivalence.
    No default path or physical gate is changed; input objects are immutable.
    """
    _, outputs = _simplification_axes(nodes, output_labels)
    protected = set(outputs)
    cache = {} if _cache is None else _cache
    identities = tuple(range(len(nodes))) if _identities is None else tuple(_identities)
    assert len(identities) == len(nodes)
    active = tuple(nodes)
    queue = list(range(len(active)))
    done: set[int] = set()
    while queue:
        identity = queue.pop()
        node = active[identity]
        key = (identities[identity], id(node.tensor))
        if key in cache:
            continue
        pair = find_tensor_diagonal(node.tensor, antidiagonal=True, atol=atol)
        if pair is None:
            cache[key] = node.tensor
            continue
        left, right = (node.labels[axis] for axis in pair)
        if left in protected:
            if right in protected:
                continue
            flipped = right
        else:
            flipped = left
        if flipped in done:
            continue
        active = _flip_edge(active, flipped)
        done.add(flipped)
        queue.append(identity)
    return active


def reduce_tensor_diagonals(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int] | None = None,
    *,
    atol: float = 1e-12,
    _cache: dict[tuple[int, int], torch.Tensor] | None = None,
    _identities: Sequence[int] | None = None,
    _adjacency: dict[int, dict[int, None]] | None = None,
) -> tuple[TensorNetworkNode, ...]:
    """Merge diagonal edges globally, then collapse the triggering tensor only.

    Follow source reverse-occurrence traversal, first-pair choice and protected
    output replacement direction. Other occurrences can acquire repeated labels
    and hyperedges, just as in the source. Existing native einsum takes the
    multidimensional diagonal. Keep the initial outputs explicit in subsequent
    passes/contraction: outer-occurrence inference can change after merging.

    Atol > 0 can discard off-diagonal contributions without a global error
    bound. At zero tolerance, values and fixed-support parameter gradients are
    preserved, but discarded zero entries' ambient derivatives need not be.
    Inputs, metadata, dtype/device and retained autograd graphs are preserved.
    This explicit D pass is not enabled in default training/register execution.
    """
    _, outputs = _simplification_axes(nodes, output_labels)
    protected = set(outputs)
    cache = {} if _cache is None else _cache
    identities = tuple(range(len(nodes))) if _identities is None else tuple(_identities)
    assert len(identities) == len(nodes)
    active = tuple(nodes)
    queue = list(range(len(active)))
    while queue:
        identity = queue.pop()
        node = active[identity]
        key = (identities[identity], id(node.tensor))
        if key in cache:
            continue
        pair = find_tensor_diagonal(node.tensor, atol=atol)
        if pair is None:
            cache[key] = node.tensor
            continue
        left, right = (node.labels[axis] for axis in pair)
        if right in protected:
            if left in protected:
                continue
            removed, retained = left, right
        else:
            removed, retained = right, left
        if _adjacency is not None and removed != retained:
            neighbors = _adjacency.pop(removed)
            kept = _adjacency.setdefault(retained, {})
            for neighbor in neighbors:
                kept.setdefault(neighbor, None)
        active = _merge_edge(active, removed, retained)
        node = active[identity]
        labels = tuple(dict.fromkeys(node.labels))
        collapsed = replace(
            node,
            tensor=einsum_reorder_by_labels(node.tensor, node.labels, labels),
            labels=labels,
        )
        active = (*active[:identity], collapsed, *active[identity + 1 :])
        queue.append(identity)
    return active
