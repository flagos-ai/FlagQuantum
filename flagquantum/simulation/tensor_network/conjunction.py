# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 And.my_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Source-ordered logical conjunction tensors, independent of physical erasure."""

from collections.abc import Mapping
from itertools import product

import torch

from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups


def conjunction_tensor_node(
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    like: torch.Tensor,
    *,
    values: tuple[int, int] = (1, 1),
    uncompute: bool = False,
) -> TensorNetworkNode:
    """Supply And's original five axes and switch only the target direction.

    Axes are ctrl-in[0], ctrl-in[1], ctrl-out[0], ctrl-out[1], target.
    Controls are two row-major one-bit words; target is one scalar bit. Forward
    creates a logical output on zero-initialized physical storage. Adjoint is
    the rectangular effect on that output, not a trace or reset channel. Actual
    measurement-assisted erasure belongs to Compiler. The caller owns labels;
    this supplier allocates no edges, gates or work. Data follows like's device
    and dtype, with the locked source's exact zero/one entries and name.
    """
    if len(values) != 2 or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("And tensor requires two concrete zero/one control values")
    if type(uncompute) is not bool:
        raise ValueError("And tensor direction must be a built-in boolean")
    bit = RegisterType("bit", 1)
    signature = Signature(
        (
            Register("ctrl", bit, (2,)),
            Register(
                "target",
                bit,
                side=RegisterSide.LEFT if uncompute else RegisterSide.RIGHT,
            ),
        )
    )
    register_tensor_label_groups(signature, incoming, outgoing)
    target = incoming["target"] if uncompute else outgoing["target"]
    labels = (
        incoming["ctrl"][0][0],
        incoming["ctrl"][1][0],
        outgoing["ctrl"][0][0],
        outgoing["ctrl"][1][0],
        target[0][0],
    )
    data = torch.zeros((2,) * 5, dtype=like.dtype, device=like.device)
    for first, second in product((0, 1), repeat=2):
        data[first, second, first, second, int((first, second) == values)] = 1
    return TensorNetworkNode(data, labels, name="And†" if uncompute else "And")
