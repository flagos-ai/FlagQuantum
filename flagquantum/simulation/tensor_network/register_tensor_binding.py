# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Axis ordering adapted from Qualtran v0.7.0 _dense.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Bind dense register tensors to existing integer-labeled contraction nodes."""

from collections import Counter
from collections.abc import Mapping, Sequence

import torch

from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .path_search import _contract_nodes_greedy
from .register_axis_groups import (
    group_register_outer_indices,
    register_tensor_label_groups,
)
from .register_tensor_indices import register_tensor_outer_indices


def bind_register_tensor(
    data: torch.Tensor,
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    *,
    name: str = "",
) -> TensorNetworkNode:
    """Reshape well-ordered dense data to binary axes and attach existing labels.

    This follows the source tensor protocol's reshape, preserving PyTorch's
    dtype, device and autograd connection. It does not copy through NumPy or
    inspect amplitudes. Invalid element counts raise ValueError. The caller
    owns connectivity, shared labels and the dense data's output/input order.
    """
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    labels = tuple(label for group in groups for label in group)
    if data.numel() != 1 << len(labels):
        raise ValueError("dense tensor size does not match its register bit axes")
    return TensorNetworkNode(data.reshape((2,) * len(labels)), labels, name=name)


def contract_register_tensors(
    nodes: Sequence[TensorNetworkNode],
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> torch.Tensor:
    """Contract existing nodes and group output/input axes to a dense result.

    Reuse the existing greedy contraction implementation; do not build a second
    executor. Pure-state outputs are scalar, vector or matrix, omitting empty
    bit groups as the original protocol does. This finite pure-state slice does
    not implement superoperator grouping, automatic component flattening, or
    a Quimb simplification policy. At least one node is required, including an
    explicit scalar node for a scalar-only network.
    """
    if not nodes:
        raise ValueError("register contraction requires at least one tensor node")
    descriptors = register_tensor_outer_indices(signature, incoming, outgoing)
    counts = Counter(label for node in nodes for label in node.labels)
    groups = group_register_outer_indices(
        signature,
        {
            index: label
            for label, index in descriptors.items()
            if isinstance(index, tuple) and counts[label] == 1
        },
    )
    labels = tuple(label for group in groups for label in group)
    result, _ = _contract_nodes_greedy(nodes, labels)
    return result.reshape(tuple(1 << len(group) for group in groups))
