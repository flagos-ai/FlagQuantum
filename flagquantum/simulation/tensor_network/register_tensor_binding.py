# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Axis ordering adapted from Qualtran v0.7.0 _dense.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Bind dense register tensors to existing integer-labeled contraction nodes."""

from collections.abc import Iterable, Mapping, Sequence

import torch

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .path_search import _contract_nodes_greedy
from .register_tensor import register_tensor_shape


def _port_labels(
    registers: Iterable[Register],
    bindings: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[int, ...]:
    labels: list[int] = []
    for register in registers:
        width = register.bitsize
        assert isinstance(width, int)
        count = len(tuple(register.all_idxs()))
        # The source never looks up connections when no bits are visited.
        if not width or not count:
            continue
        words = bindings[register.name]
        if len(words) != count or any(len(word) != width for word in words):
            raise ValueError("tensor bindings must match register element/bit counts")
        for word in words:
            for label in word:
                if type(label) is not int or label < 0:
                    raise ValueError("tensor labels must be nonnegative integers")
                labels.append(label)
    return tuple(labels)


def register_tensor_label_groups(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[tuple[int, ...], ...]:
    """Order nonempty output and input bit groups as the pinned dense protocol.

    Each port maps to one tuple of bit labels per row-major word element, even
    for a scalar word. Labels refer to the caller's existing tensor graph, not
    physical wire ownership. Zero-width/empty ports require no mapping. Extra
    keys are ignored, while missing live ports raise KeyError. No labels are
    allocated, renamed or deduplicated here.
    """
    register_tensor_shape(signature)
    right = _port_labels(signature.rights(), outgoing)
    left = _port_labels(signature.lefts(), incoming)
    return tuple(group for group in (right, left) if group)


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
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    if not nodes:
        raise ValueError("register contraction requires at least one tensor node")
    labels = tuple(label for group in groups for label in group)
    result, _ = _contract_nodes_greedy(nodes, labels)
    return result.reshape(tuple(1 << len(group) for group in groups))
