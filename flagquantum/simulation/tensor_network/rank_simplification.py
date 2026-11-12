# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Rank-reduction traversal adapted from quimb 1.15.0 TensorNetwork.rank_simplify,
# called by Qualtran v0.7.0 simulation/tensor/_dense.py::quimb_to_dense.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/tensor_core.py

"""Exact shape-driven rank reduction over native tensor nodes and kernels."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations
from math import isqrt

import torch

from .models import TensorNetworkNode
from .simplification_axes import _simplification_axes
from .stages import einsum_pair_by_labels, einsum_reorder_by_labels


def simplify_tensor_ranks(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int] | None = None,
    *,
    max_combinations: int = 500,
) -> tuple[TensorNetworkNode, ...]:
    """Contract connected pairs only when the result's rank does not increase.

    Preserve protected output labels, hyperedges and repeated-index diagonals.
    Pair selection follows the source index queue, rank gain and occurrence-ID
    tie breaks; existing native einsum kernels own numerical execution. The
    input nodes are immutable, and dtype/device/autograd are preserved. This
    pass has no amplitude tolerance, SVD, normalization or physical operation.
    Meta tensors are supported because all decisions depend only on shape.

    Floating scalars are multiplied into the first surviving node, rather than
    spread through fractional powers as in Quimb. This preserves finite zero
    gradients and makes scalar ownership explicit; scalar-bearing leaf arrays
    need not match the source representation, but their full contraction does.
    Empty networks, missing/duplicate outputs, inconsistent dimensions, mixed
    dtype/device and invalid search budgets raise ValueError. This is the R
    dependency, not a replacement for the source's complete ADCR policy.
    """
    dims, outputs = _simplification_axes(nodes, output_labels)
    if type(max_combinations) is not int or max_combinations < 1:
        raise ValueError("rank simplification requires a positive integer budget")
    if any(
        node.tensor.dtype != nodes[0].tensor.dtype
        or node.tensor.device != nodes[0].tensor.device
        for node in nodes
    ):
        raise ValueError("rank simplification requires one dtype and device")

    active: dict[int, TensorNetworkNode] = {}
    scalars: list[torch.Tensor] = []
    for identity, node in enumerate(nodes):
        if not node.labels:
            scalars.append(node.tensor)
            continue
        labels = tuple(dict.fromkeys(node.labels))
        active[identity] = (
            node
            if labels == node.labels
            else replace(
                node,
                tensor=einsum_reorder_by_labels(node.tensor, node.labels, labels),
                labels=labels,
            )
        )
    counts = Counter(label for node in active.values() for label in node.labels)
    counts.update(outputs)
    trivial = len(counts) == 1
    queue = dict.fromkeys(
        sorted(
            counts,
            key=lambda label: (
                dims[label],
                -sum(
                    len(node.labels) for node in active.values() if label in node.labels
                ),
            ),
        )
    )
    cutoff = (1 + isqrt(8 * max_combinations + 1)) // 2
    failed_pairs: set[tuple[int, int, int, int]] = set()
    next_identity = len(nodes)
    while queue:
        label, _ = queue.popitem()
        connected = [i for i, node in active.items() if label in node.labels]
        if not connected:
            continue
        if counts[label] == 1:
            identity = connected[0]
            node = active[identity]
            labels = tuple(value for value in node.labels if value != label)
            tensor = einsum_reorder_by_labels(node.tensor, node.labels, labels)
            if labels:
                active[identity] = replace(node, tensor=tensor, labels=labels)
            else:
                active.pop(identity)
                scalars.append(tensor)
            continue
        if len(connected) > cutoff:
            connected.sort(key=lambda identity: len(active[identity].labels))
        candidates: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = []
        checked = 0
        for left, right in combinations(connected, 2):
            a, b = active[left], active[right]
            cache_key = (left, right, id(a.tensor), id(b.tensor))
            if cache_key in failed_pairs:
                continue
            checked += 1
            involved = Counter((*a.labels, *b.labels))
            retained = tuple(
                key for key, count in involved.items() if count != counts[key]
            )
            shared = tuple(key for key in retained if involved[key] == 2)
            gain = len(retained) - max(len(a.labels), len(b.labels))
            if gain <= 0:
                candidates.append((gain, left, right, retained, shared))
            else:
                failed_pairs.add(cache_key)
            if candidates and (trivial or checked > max_combinations):
                break
        if not candidates:
            continue
        _, left, right, retained, shared = min(candidates)
        a, b = active.pop(left), active.pop(right)
        tensor = einsum_pair_by_labels(a.tensor, a.labels, b.tensor, b.labels, retained)
        for key in shared:
            counts[key] -= 1
        if retained:
            active[next_identity] = TensorNetworkNode(tensor, retained)
            next_identity += 1
            for key in retained:
                queue.setdefault(key, None)
        else:
            scalars.append(tensor)

    if scalars:
        factor = scalars[0]
        for scalar in scalars[1:]:
            factor = factor * scalar
        if active:
            identity = next(iter(active))
            node = active[identity]
            active[identity] = replace(node, tensor=node.tensor * factor)
        else:
            return (TensorNetworkNode(factor, ()),)
    return tuple(active.values())
