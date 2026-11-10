# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_forward_tensor and make_backward_tensor.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Double existing tensor nodes into independent forward/conjugate networks."""

from collections.abc import Sequence
from dataclasses import replace

from .models import TensorNetworkNode


def conjugate_pair_labels(
    labels: tuple[int, ...], *, backward: bool = False
) -> tuple[int, ...]:
    """Encode existing labels as even forward or odd backward integers.

    The mapping is injective across both directions, including sparse labels.
    It does not transpose axes or allocate physical wires. Invalid labels and
    non-boolean direction flags raise ValueError. Repeated labels are preserved
    because they may describe existing repeated-index contractions.
    """
    if type(backward) is not bool:
        raise ValueError("conjugate direction requires a built-in boolean")
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("conjugate pair labels must be nonnegative integers")
    return tuple(2 * label + int(backward) for label in labels)


def tensor_conjugate_pairs(
    nodes: Sequence[TensorNetworkNode],
) -> tuple[TensorNetworkNode, ...]:
    """Return interleaved forward and elementwise conjugate copies.

    Like source Quimb Tensor.H, backward data is conjugated without an axis
    permutation. Forward data shares the original storage; PyTorch owns the
    conjugate view and autograd. Dtype, device, name and metadata are retained.
    Nodes and their metadata are not mutated. These copies describe a numerical
    outer product, not inverse circuit execution or additional quantum gates.

    Rank/label mismatches and invalid labels raise ValueError before producing
    the result. Empty input returns empty output, not a scalar identity.
    This does not implement discard/trace, Kraus sums, external grouping,
    graph expansion or an execution backend.
    """
    for node in nodes:
        if len(node.labels) != node.tensor.ndim:
            raise ValueError("conjugate tensor rank must match its labels")
        conjugate_pair_labels(node.labels)
    return tuple(
        copy
        for node in nodes
        for copy in (
            replace(node, labels=conjugate_pair_labels(node.labels)),
            replace(
                node,
                tensor=node.tensor.conj(),
                labels=conjugate_pair_labels(node.labels, backward=True),
            ),
        )
    )
