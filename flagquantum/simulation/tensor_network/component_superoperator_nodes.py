# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cbloq_to_superquimb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Wire mixed component channels through the existing SSA and trace machinery."""

from collections.abc import Callable, Iterable, Mapping

import torch

from ...core.ir.program import FunctionRecord, ProgramRecord
from .component_tensor_nodes import _component_tensor_nodes
from .conjugate_pairs import tensor_conjugate_pairs
from .models import TensorNetworkNode
from .trace_indices import trace_conjugate_indices


def component_superoperator_nodes(
    record: ProgramRecord,
    order: tuple[int, ...],
    supply: Callable[
        [
            FunctionRecord,
            Mapping[str, tuple[tuple[int, ...], ...]],
            Mapping[str, tuple[tuple[int, ...], ...]],
            Callable[[], int],
        ],
        Iterable[TensorNetworkNode | int],
    ],
    like: torch.Tensor,
    entry: str = "main",
) -> tuple[
    tuple[TensorNetworkNode, ...],
    dict[str, tuple[tuple[int, ...], ...]],
    dict[str, tuple[tuple[int, ...], ...]],
]:
    """Return a doubled mixed quantum/classical graph with explicit traces.

    The supplier receives a selected typed callee, row-major/MSB-first port maps,
    and a fresh-label callable sharing the SSA wiring namespace. Call it during
    supply for hidden environment edges; never infer a fresh label from local
    ports. Yield numerical nodes or original integer labels to discard, matching
    the source tensor/DiscardInd protocol without foreign objects. Source-like
    measurement suppliers yield their environment label after their copy node.

    Reuse the pure graph's validation, visitation and unused-port identities,
    then conjugate-double and trace the explicitly supplied labels. Returned
    port maps remain undoubled for contract_register_superoperator. Supplier
    failures propagate; trace validation rejects missing/internal/duplicate legs.
    No normalization, sampling, physical allocation, automatic leaf discovery,
    backend selection or compiler traversal occurs. Pure-state wiring retains
    its existing classical-port refusal and three-argument supplier interface.
    """
    discarded: list[int] = []

    def attach(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
        fresh_label: Callable[[], int],
    ) -> Iterable[TensorNetworkNode]:
        for item in supply(function, incoming, outgoing, fresh_label):
            if isinstance(item, TensorNetworkNode):
                yield item
            else:
                discarded.append(item)

    nodes, incoming, outgoing = _component_tensor_nodes(
        record, order, attach, like, entry, classical=True
    )
    external = {
        label
        for bindings in (incoming, outgoing)
        for words in bindings.values()
        for word in words
        for label in word
    }
    if external.intersection(discarded):
        raise ValueError("discarded indices must not be external ports")
    doubled = tensor_conjugate_pairs(nodes)
    return trace_conjugate_indices(doubled, tuple(discarded)), incoming, outgoing
