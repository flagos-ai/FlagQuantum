# Copyright 2023 Google LLC
# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Shape-only delegation follows Qualtran v0.7.0 _dense.py and Quimb 1.15.0
# TensorNetwork.contraction_width/contraction_tree, using existing optional Cotengra.

"""Return the locked dense protocol's contraction width without numerical work."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from .models import TensorNetworkNode
from .path_search import _label_dims


def tensor_contraction_width(nodes: Sequence[TensorNetworkNode]) -> float:
    """Return the locked-default Cotengra greedy tree's reported width.

    Match the locked Quimb default's shape-only call to array_contract_tree,
    including output inference and canonicalization. This is not native greedy
    path search, a memory-capacity guarantee or a measured RAM peak. Scalar,
    repeated-index and hyperedge handling remain with that same planner. No
    numerical arrays, external trees or paths cross this boundary.
    Cotengra0.8.3 returns raw size for a single leaf before applying log2; that
    source defect is preserved and tested rather than silently changing the
    simplification trigger. Multi-leaf width uses log2(maximum produced size).

    The existing optional Cotengra package is required; there is no fallback.
    Meta tensors are supported. Empty networks, inconsistent dimensions and
    zero extents are refused; an invalid planner result raises ValueError.
    Upstream process/thread strategy overrides are not mirrored: this helper
    implements the pinned default strategy, explicitly greedy.
    """
    if not nodes:
        raise ValueError("contraction width requires at least one tensor node")
    dims = _label_dims(nodes)
    if any(size < 1 for size in dims.values()):
        raise ValueError("contraction width requires positive index dimensions")
    try:
        import cotengra as ctg
    except ImportError as error:
        raise ImportError(
            "source-aligned width planning requires the optional 'cotengra' package"
        ) from error
    tree = ctg.array_contract_tree(
        tuple(node.labels for node in nodes),
        output=None,
        shapes=tuple(tuple(node.tensor.shape) for node in nodes),
        optimize="greedy",
    )
    width = tree.contraction_width()
    if (
        isinstance(width, bool)
        or not isinstance(width, Real)
        or not isfinite(width)
        or width < 0
    ):
        raise ValueError("contraction planner must return a finite nonnegative width")
    return float(width)
