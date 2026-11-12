# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Default control loop adapted from quimb 1.15.0 TensorNetwork.full_simplify.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/tensor_core.py

"""Explicit default ADCR fixed point over existing native simplification owners."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real

import torch

from .column_reduction import reduce_tensor_columns
from .contraction_width import tensor_contraction_width
from .diagonal_reduction import gauge_tensor_antidiagonals, reduce_tensor_diagonals
from .models import TensorNetworkNode
from .rank_simplification import simplify_tensor_ranks
from .simplification_axes import _simplification_axes, squeeze_tensor_axes


def simplify_tensor_network(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int] | None = None,
    *,
    atol: float = 1e-12,
) -> tuple[TensorNetworkNode, ...]:
    """Squeeze once, then repeat ADCR until tensor/index counts stabilize.

    Outputs are captured before squeezing. Index count is the number of distinct
    labels, not axis occurrences; always execute one full round. Reuse cached
    negative checks and occurrence identities across rounds as the reference
    default policy does. Cache values retain tensors so Python object IDs cannot
    be recycled during the run, and all cache storage is released on return.

    This explicitly requested numerical preprocessing can discard contributions
    at positive tolerance and ambient zero-entry derivatives even at atol=0.
    Fixed-support parameter gradients are retained. It is not automatically
    applied by register consumers or training, and does not estimate contraction
    width. Existing pass-specific source differences remain, including native
    repeated-edge A gauging and first-leaf scalar absorption in R. Norm balancing,
    custom sequences, S/L/P splitting and external persistent caches are outside
    the default ADCR slice. Inputs are immutable; no physical gates are emitted.
    """
    _, outputs = _simplification_axes(nodes, output_labels)
    active = squeeze_tensor_axes(nodes, outputs)
    identities = list(range(len(active)))
    a_cache: dict[tuple[int, int], torch.Tensor] = {}
    d_cache: dict[tuple[int, int], torch.Tensor] = {}
    c_cache: dict[tuple[int, int], torch.Tensor] = {}
    r_cache: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    adjacency: dict[int, dict[int, None]] = {}
    for identity, node in enumerate(active):
        for label in node.labels:
            adjacency.setdefault(label, {})[identity] = None
    previous = (-1, -1)
    counts = (len(active), len({label for node in active for label in node.labels}))
    while counts != previous:
        active = gauge_tensor_antidiagonals(
            active, outputs, atol=atol, _cache=a_cache, _identities=identities
        )
        active = reduce_tensor_diagonals(
            active,
            outputs,
            atol=atol,
            _cache=d_cache,
            _identities=identities,
            _adjacency=adjacency,
        )
        active = reduce_tensor_columns(
            active, outputs, atol=atol, _cache=c_cache, _identities=identities
        )
        present = {label for node in active for label in node.labels}
        for label in tuple(adjacency):
            if label not in present:
                del adjacency[label]
        active = simplify_tensor_ranks(
            active,
            outputs,
            _cache=r_cache,
            _identities=identities,
            _adjacency=adjacency,
        )
        previous = counts
        counts = (len(active), len({label for node in active for label in node.labels}))
    return active


def simplify_wide_tensor_network(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    atol: float,
) -> tuple[TensorNetworkNode, ...]:
    """Apply default ADCR only when the locked-default planner width exceeds 12.

    This explicit entry accepts support-dependent simplification and its
    tolerance/ambient-gradient boundaries. Width comes from the same optional
    Cotengra planner as the locked source, without inspecting numerical values.
    Capture register output groups before calling. No physical operation,
    normalization or default training policy is introduced.
    """
    if (
        isinstance(atol, bool)
        or not isinstance(atol, Real)
        or not isfinite(atol)
        or atol < 0
    ):
        raise ValueError("structure tolerance must be a finite nonnegative real number")
    _, outputs = _simplification_axes(nodes, output_labels)
    if tensor_contraction_width(nodes) > 12:
        return simplify_tensor_network(nodes, outputs, atol=atol)
    return tuple(nodes)
