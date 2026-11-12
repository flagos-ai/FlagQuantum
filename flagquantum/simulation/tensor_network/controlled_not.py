# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CNOT.my_tensors, COPY and XOR.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/cnot.py

"""Factor a coherent controlled-NOT into source-ordered COPY and XOR leaves."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import product

import torch

from ...core.ir.program.signatures import Signature
from .measurement_copy import z_measurement_copy_node
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups


def controlled_not_nodes(
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    fresh_label: Callable[[], int],
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, TensorNetworkNode]:
    """Return exact COPY/XOR data and source axis order on four quantum ports.

    Ports are scalar one-bit ctrl/target THRU registers. COPY axes are
    (ctrl-in, ctrl-out, internal), XOR axes (target-in, target-out, internal).
    One fresh numerical edge comes from the caller's graph allocator. External
    and internal labels must be distinct nonnegative built-in integers.
    Dtype/device follow like. The shared Z-copy data is a coherent equality
    tensor here: no environment is traced and no measurement occurs.
    No dense 4x4 matrix, physical gate, work bit or classical slot is created.
    """
    groups = register_tensor_label_groups(
        Signature.build(ctrl=1, target=1), incoming, outgoing
    )
    labels = tuple(label for group in groups for label in group)
    if len(set(labels)) != 4:
        raise ValueError("controlled-NOT external tensor labels must be distinct")
    internal = fresh_label()
    if type(internal) is not int or internal < 0 or internal in labels:
        raise ValueError(
            "controlled-NOT internal tensor label must be fresh and nonnegative"
        )
    copy = replace(
        z_measurement_copy_node(
            incoming["ctrl"][0][0], outgoing["ctrl"][0][0], internal, like
        ),
        name="COPY",
    )
    bits = torch.tensor(list(product((0, 1), repeat=3)), device=like.device)
    xor = (1 - bits.sum(dim=1) % 2).reshape(2, 2, 2).to(dtype=like.dtype)
    return copy, TensorNetworkNode(
        xor, (incoming["target"][0][0], outgoing["target"][0][0], internal), name="XOR"
    )
