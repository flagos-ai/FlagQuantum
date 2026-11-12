# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Output protection and squeezing follow quimb 1.15.0 TensorNetwork.full_simplify.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/tensor_core.py

"""Shared output protection and shape-only squeezing for numerical passes."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from .models import TensorNetworkNode
from .path_search import _label_dims


def _simplification_axes(
    nodes: Sequence[TensorNetworkNode], output_labels: Sequence[int] | None
) -> tuple[dict[int, int], tuple[int, ...]]:
    if not nodes:
        raise ValueError("tensor simplification requires at least one tensor node")
    dims = _label_dims(nodes)
    counts = Counter(label for node in nodes for label in node.labels)
    outputs = (
        tuple(label for label, count in counts.items() if count == 1)
        if output_labels is None
        else tuple(output_labels)
    )
    if len(set(outputs)) != len(outputs) or any(label not in dims for label in outputs):
        raise ValueError(
            "tensor simplification outputs must be distinct existing labels"
        )
    return dims, outputs


def squeeze_tensor_axes(
    nodes: Sequence[TensorNetworkNode], output_labels: Sequence[int] | None = None
) -> tuple[TensorNetworkNode, ...]:
    """Remove unprotected size-one axes without absorbing scalars or outputs.

    Outputs default to labels occurring once before transformation, matching
    the full-simplification entry policy. Explicit outputs also protect shared
    indices. Preserve empty dimensions, repeated labels, dtype/device, metadata,
    autograd and input objects; unchanged nodes retain identity. Meta tensors
    work because selection is shape-only. No scalar normalization or output
    reshaping occurs. Invalid networks/outputs fail through shared validation.
    """
    _, outputs = _simplification_axes(nodes, output_labels)
    protected = set(outputs)
    result = []
    for node in nodes:
        axes = tuple(
            axis
            for axis, (label, size) in enumerate(
                zip(node.labels, node.tensor.shape, strict=True)
            )
            if size == 1 and label not in protected
        )
        result.append(
            replace(
                node,
                tensor=node.tensor.squeeze(axes),
                labels=tuple(
                    label for axis, label in enumerate(node.labels) if axis not in axes
                ),
            )
            if axes
            else node
        )
    return tuple(result)
