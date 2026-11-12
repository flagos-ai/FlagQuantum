# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _quimb._get_placeholder_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Preserve unused direct-through bit connections in existing tensor networks."""

from collections.abc import Mapping

import torch

from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups


def register_identity_nodes(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, ...]:
    """Return one binary identity node per caller-declared direct-through bit.

    The signature contains only unused THRU ports, not all ports of a program.
    The caller determines direct connectivity; these nodes must not also be
    inserted for operated ports. Bindings follow the existing row-major word,
    MSB-first bit protocol. Their external labels must be distinct, including
    between input and output. No graph label is allocated or renamed here.

    Each node's axes are output then input, with dtype/device taken from like.
    Zero-width and empty-array signatures return no nodes, matching the source
    bit loop; scalar-network handling remains the caller's responsibility.
    Storage is linear in bit count, with no full word identity materialization.
    Invalid sides, labels or bindings raise ValueError/KeyError before nodes
    are allocated. The helper adds no gate, reset, measurement or workspace.
    """
    if not signature.thru_registers_only:
        raise ValueError("identity placeholders require direct-through ports")
    return register_bit_wiring_nodes(signature, incoming, outgoing, like)


def register_bit_wiring_nodes(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, ...]:
    """Factor a declared bit-preserving rebind into binary identities.

    Unlike direct-through placeholders, port names, widths and shapes may differ
    across sides. Signature order supplies the explicit bit correspondence:
    row-major words, then MSB-first bits. This is a wiring promise supplied by
    the caller, not inference from a component name or body. Unequal bit counts
    or reused labels are refused. Empty wiring emits no nodes.
    No whole-word matrix, physical gate or work allocation is created.
    """
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    if not groups:
        return ()
    if len(groups) != 2 or len(groups[0]) != len(groups[1]):
        raise ValueError("bit wiring requires equal input and output bit counts")
    right, left = groups
    labels = (*right, *left)
    if len(set(labels)) != len(labels):
        raise ValueError("identity placeholder external labels must be distinct")
    return tuple(
        TensorNetworkNode(
            torch.eye(2, dtype=like.dtype, device=like.device),
            (output, input_),
            name="register_identity",
        )
        for output, input_ in zip(right, left, strict=True)
    )
