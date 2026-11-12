# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Axis ordering adapted from Qualtran v0.7.0 _dense.py::_group_outer_inds.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Group open-system register axes on the existing numerical tensor network."""

from collections.abc import Mapping, Sequence

import torch

from ...core.ir.program.signatures import Signature
from .conjugate_pairs import conjugate_pair_labels
from .full_simplification import simplify_wide_tensor_network
from .models import TensorNetworkNode
from .path_search import _contract_nodes_greedy
from .register_axis_groups import (
    register_tensor_label_groups,
)
from .register_tensor_indices import register_contraction_groups


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
    *,
    output_groups: tuple[tuple[int, ...], ...] | None = None,
    simplification_atol: float | None = None,
) -> torch.Tensor:
    """Contract doubled nodes as a scalar, density/effect matrix, or channel.

    Nodes already contain conjugate-pair labels and any explicit trace merges.
    Capture register_contraction_groups before explicit simplification and pass
    output_groups here to retain protected outputs that become hyperedges.
    A non-None simplification_atol explicitly enables the source-default width
    >12 ADCR policy after capturing groups. It requires optional Cotengra and
    accepts support-dependent tolerance/ambient-gradient changes. None retains
    the existing unsimplified path, including arbitrary trainable entries.
    Reuse the native greedy contraction engine without normalization, discard
    discovery or reset. At least one node is required,
    including an explicit scalar node for an all-scalar network. PyTorch retains
    dtype, device and autograd. Dense channel storage grows as 16**n for n bits
    on each side; this helper does not provide a memory-capacity guarantee.
    """
    if not nodes:
        raise ValueError("register contraction requires at least one tensor node")
    groups = register_contraction_groups(
        nodes,
        signature,
        incoming,
        outgoing,
        superoperator=True,
        output_groups=output_groups,
    )
    if simplification_atol is not None:
        nodes = simplify_wide_tensor_network(
            nodes,
            tuple(label for group in groups for label in group),
            atol=simplification_atol,
        )
    result, _ = _contract_nodes_greedy(
        nodes, tuple(label for group in groups for label in group)
    )
    return result.reshape(tuple(1 << len(group) for group in groups))
