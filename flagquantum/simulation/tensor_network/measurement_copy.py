# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MeasureZ.my_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Supply the numerical copy leaf of a computational-basis measurement."""

import torch

from .models import TensorNetworkNode


def z_measurement_copy_node(
    input_label: int,
    result_label: int,
    environment_label: int,
    like: torch.Tensor,
) -> TensorNetworkNode:
    """Tie input, classical result and environment in source axis order.

    The caller owns three distinct nonnegative integer edges and must trace the
    environment after conjugate doubling to obtain a measurement channel.
    Keeping the result exposes its classical record; explicitly tracing it
    discards that record. This leaf alone is a coherent copy, not a sampled or
    normalized measurement. Dtype/device follow like; no physical instruction,
    environment allocation, reset, graph traversal or random label is emitted.
    Invalid edge labels raise ValueError before constructing tensor data.
    """
    labels = (input_label, result_label, environment_label)
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("measurement copy labels must be nonnegative integers")
    if len(set(labels)) != 3:
        raise ValueError("measurement copy labels must be distinct")
    data = torch.zeros((2, 2, 2), dtype=like.dtype, device=like.device)
    data[0, 0, 0] = data[1, 1, 1] = 1
    return TensorNetworkNode(data, labels, name="z_measurement_copy")
