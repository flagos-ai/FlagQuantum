# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Traversal adapted from quimb 1.15.0 TensorNetwork.column_reduce.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/tensor_core.py

"""Explicit support-dependent column reduction on existing native tensor nodes."""

from collections.abc import Sequence

import torch

from .contraction import _slice_nodes
from .models import TensorNetworkNode
from .simplification_axes import _simplification_axes
from .tensor_columns import find_tensor_column


def reduce_tensor_columns(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int] | None = None,
    *,
    atol: float = 1e-12,
    _cache: dict[tuple[int, int], torch.Tensor] | None = None,
    _identities: Sequence[int] | None = None,
) -> tuple[TensorNetworkNode, ...]:
    """Cut internally supported columns in source reverse-occurrence order.

    Probe the first candidate axis only: if it is a protected output, stop
    processing that tensor rather than search later axes. Slice every occurrence
    of an internal label, including hyperedges and repeated axes, through the
    existing native slicing owner. Requeue only the triggering occurrence, as
    the source does; already-processed neighbors wait for a later external pass.

    Input objects/data are never modified. Retained slices preserve autograd,
    dtype/device and metadata. Support choices are data dependent: atol > 0 can
    discard small contributions with no global contraction-error bound. Even at
    atol=0 the ambient Jacobian of discarded zero entries is not retained;
    gradients agree on a fixed sparse parameterization, not topology changes.
    This explicit C pass is not enabled in default register or training paths.
    Invalid outputs or unsupported detector values fail without partial writes.
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
        candidate = find_tensor_column(node.tensor, atol=atol)
        if candidate is None:
            cache[key] = node.tensor
            continue
        axis, column = candidate
        label = node.labels[axis]
        if label in protected:
            continue
        active = _slice_nodes(active, {label: column})
        queue.append(identity)
    return active
