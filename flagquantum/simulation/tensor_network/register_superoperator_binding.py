# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Axis ordering adapted from Qualtran v0.7.0 _dense.py::_group_outer_inds.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Group open-system register axes on the existing numerical tensor network."""

from collections.abc import Mapping, Sequence

import torch

from ...core.ir.program.signatures import Signature
from .conjugate_pairs import conjugate_pair_labels
from .models import TensorNetworkNode
from .path_search import _contract_nodes_greedy
from .register_tensor_binding import register_tensor_label_groups


def register_superoperator_label_groups(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[tuple[int, ...], ...]:
    """Order nonempty right-forward, right-backward, left-forward, left-backward.

    Bindings contain original, undoubled integer labels, one MSB-first word per
    row-major register element. Quantum and classical bits both contribute axes.
    The existing register binder validates widths and bindings; zero-width and
    empty registers need no connections. No graph or physical wires are created.
    """
    return tuple(
        conjugate_pair_labels(group, backward=backward)
        for group in register_tensor_label_groups(signature, incoming, outgoing)
        for backward in (False, True)
    )


def contract_register_superoperator(
    nodes: Sequence[TensorNetworkNode],
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> torch.Tensor:
    """Contract doubled nodes as a scalar, density/effect matrix, or channel.

    Nodes already contain conjugate-pair labels and any explicit trace merges.
    Reuse the native greedy contraction engine without normalization, discard
    discovery, reset, or Quimb simplification. At least one node is required,
    including an explicit scalar node for an all-scalar network. PyTorch retains
    dtype, device and autograd. Dense channel storage grows as 16**n for n bits
    on each side; this helper does not provide a memory-capacity guarantee.
    """
    groups = register_superoperator_label_groups(signature, incoming, outgoing)
    if not nodes:
        raise ValueError("register contraction requires at least one tensor node")
    result, _ = _contract_nodes_greedy(
        nodes, tuple(label for group in groups for label in group)
    )
    return result.reshape(tuple(1 << len(group) for group in groups))
