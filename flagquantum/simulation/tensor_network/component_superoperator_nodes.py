# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cbloq_to_superquimb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Wire mixed component channels through the existing SSA and trace machinery."""

from collections.abc import Callable, Iterable, Mapping

import torch

from ...core.ir.program import FunctionRecord, ProgramRecord
from .component_tensor_nodes import _component_tensor_nodes
from .conjugate_pairs import conjugate_pair_labels, tensor_conjugate_pairs
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
    A discarded external input with no numerical leg gets a binary identity
    effect on its forward/backward input axes. This extends the source's direct
    input-discard case, whose empty network fails contraction. Declared input
    maps remain intact; discarding an external output or an input already used
    by a numerical tensor is refused. Each effect shares one four-element
    tensor with other input effects, with dtype/device taken from like.
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
    conjugate_pair_labels(tuple(discarded))
    if len(discarded) != len(set(discarded)):
        raise ValueError("trace requests must name distinct original indices")
    inputs = {label for words in incoming.values() for word in words for label in word}
    outputs = {label for words in outgoing.values() for word in words for label in word}
    if outputs.intersection(discarded):
        raise ValueError("discarded indices must not be external output ports")
    trace_only = tuple(label for label in discarded if label in inputs)
    if any(label in node.labels for node in nodes for label in trace_only):
        raise ValueError(
            "trace-only external inputs must have no numerical tensor legs"
        )
    internal = tuple(label for label in discarded if label not in inputs)
    doubled = trace_conjugate_indices(tensor_conjugate_pairs(nodes), internal)
    # A direct external discard has no producer tensor to reindex. Retain its
    # two input axes with a trace effect instead of removing the declared port.
    if trace_only:
        identity = torch.eye(2, dtype=like.dtype, device=like.device)
        doubled += tuple(
            TensorNetworkNode(
                identity,
                (
                    *conjugate_pair_labels((label,)),
                    *conjugate_pair_labels((label,), backward=True),
                ),
                name="input_trace_effect",
            )
            for label in trace_only
        )
    return doubled, incoming, outgoing
