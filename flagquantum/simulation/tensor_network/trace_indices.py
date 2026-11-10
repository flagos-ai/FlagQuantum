# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Index merging adapted from Qualtran v0.7.0 cbloq_to_superquimb DiscardInd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Trace exposed forward/conjugate edges through existing numerical labels."""

from collections.abc import Sequence
from dataclasses import replace

from .conjugate_pairs import conjugate_pair_labels
from .models import TensorNetworkNode


def trace_conjugate_indices(
    nodes: Sequence[TensorNetworkNode], labels: tuple[int, ...]
) -> tuple[TensorNetworkNode, ...]:
    """Join specified original indices across a doubled numerical network.

    Nodes use GRAPH62 even forward/odd backward labels. The caller identifies
    exposed legs to discard using original undoubled labels. Replacing each
    backward label by its forward partner makes the existing contraction sum
    both legs over a shared index, matching source DiscardInd reindexing.
    No tensor data, axes, dtype, device, metadata or gradient is changed.

    Each requested leg must occur once in each direction with equal dimension.
    Invalid/repeated requests, missing/already-internal legs, incompatible
    dimensions and node rank mismatches raise ValueError before any replacement.
    These stricter native checks prevent silently discarding an invalid edge.
    Empty requests retain the existing network and empty input traces nothing.
    No normalization, projection, reset, measurement, classical ownership or
    graph traversal is inferred. Consumers must omit traced legs from outputs.
    """
    forward = conjugate_pair_labels(labels)
    backward = conjugate_pair_labels(labels, backward=True)
    if len(labels) != len(set(labels)):
        raise ValueError("trace requests must name distinct original indices")
    dimensions: dict[int, list[int]] = {edge: [] for edge in (*forward, *backward)}
    for node in nodes:
        if len(node.labels) != node.tensor.ndim:
            raise ValueError("trace tensor rank must match its labels")
        conjugate_pair_labels(node.labels)
        for edge, size in zip(node.labels, node.tensor.shape, strict=True):
            if edge in dimensions:
                dimensions[edge].append(size)
    for first, second in zip(forward, backward, strict=True):
        if len(dimensions[first]) != 1 or len(dimensions[second]) != 1:
            raise ValueError("trace requires one exposed edge in each direction")
        if dimensions[first] != dimensions[second]:
            raise ValueError("trace forward/backward dimensions must match")
    replacements = dict(zip(backward, forward, strict=True))
    result = []
    for node in nodes:
        mapped = tuple(replacements.get(edge, edge) for edge in node.labels)
        result.append(replace(node, labels=mapped) if mapped != node.labels else node)
    return tuple(result)
