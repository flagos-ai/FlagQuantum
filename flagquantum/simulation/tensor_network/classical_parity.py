# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Value semantics adapted from Qualtran v0.7.0 Parity.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Factor the classical parity truth table onto caller-owned numerical edges."""

from collections.abc import Callable, Mapping

import torch

from ...core.ir.program.register_types import classical_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .register_tensor_binding import register_tensor_label_groups


def classical_parity_nodes(
    count: int,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    fresh_label: Callable[[], int],
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, ...]:
    """Preserve x and output sum(x) & 1 using a linear binary tensor chain.

    Bindings follow the source x:CBit[count] THRU, parity:CBit RIGHT signature.
    Request count internal labels from the owning graph allocator. Each of the
    count four-axis nodes has axes (x-out, carry-out, x-in, carry-in), followed
    by one zero-carry ket. Empty x yields only the parity-zero ket. Labels are
    distinct nonnegative integers, including all allocated internal labels.

    This is a native factorization of the original classical value mapping,
    not a port of source my_tensors: locked Parity has no tensor implementation.
    No input dephasing is inferred. The mixed channel caller owns measurement
    environments and explicit discard. Dtype/device follow like; all nodes
    have 16*count+2 logical elements, with shared step-tensor storage.
    No physical XOR, gate, workspace, reset or sampling is emitted.
    """
    if type(count) is not int or count < 0:
        raise ValueError("parity tensors require a built-in nonnegative count")
    signature = Signature(
        (
            Register("x", classical_bit_type(), (count,)),
            Register("parity", classical_bit_type(), side=RegisterSide.RIGHT),
        )
    )
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    labels = tuple(label for group in groups for label in group)
    if len(set(labels)) != len(labels):
        raise ValueError("parity tensor external labels must be distinct")
    internal = tuple(fresh_label() for _ in range(count))
    if any(type(label) is not int or label < 0 for label in internal):
        raise ValueError("parity tensor internal labels must be nonnegative integers")
    if len(set((*labels, *internal))) != len(labels) + count:
        raise ValueError("parity tensor internal labels must be fresh and distinct")
    result = outgoing["parity"][0][0]
    chain = (*internal, result)
    nodes: list[TensorNetworkNode] = []
    if count:
        data = torch.zeros((2, 2, 2, 2), dtype=like.dtype, device=like.device)
        for bit in (0, 1):
            for carry in (0, 1):
                data[bit, bit ^ carry, bit, carry] = 1
        nodes.extend(
            TensorNetworkNode(
                data,
                (outgoing["x"][i][0], chain[i + 1], incoming["x"][i][0], chain[i]),
                name="classical_parity_step",
            )
            for i in range(count)
        )
    nodes.append(
        TensorNetworkNode(
            torch.tensor([1, 0], dtype=like.dtype, device=like.device),
            (chain[0],),
            name="classical_parity_zero",
        )
    )
    return tuple(nodes)
