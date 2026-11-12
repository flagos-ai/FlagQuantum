# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Zero-state tensors adapted from Qualtran v0.7.0 ZeroState.my_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Supply independent zero kets for already bound pure-output register bits."""

from collections.abc import Mapping

import torch

from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .register_basis_vectors import register_basis_vector_nodes


def register_zero_state_nodes(
    signature: Signature,
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, ...]:
    """Return binary |0> nodes in output word/bit order without dense words.

    Only pure-output quantum ports are accepted. The caller owns graph labels
    and the promise that these nodes implement the selected zero-state leaf.
    Dtype/device follow like. Empty arrays and zero-width words emit no nodes;
    they do not imply a scalar tensor or an automatic empty-network identity.
    Invalid sides, classical ports and repeated bit labels raise ValueError.
    Existing register binding validation supplies shape and missing-key errors.
    This supplies numerical kets, not physical allocation or reset instructions.
    """
    if tuple(signature.lefts()) or any(r.dtype.classical_bits for r in signature):
        raise ValueError("zero-state tensors require pure-output quantum ports")
    return register_basis_vector_nodes(signature, outgoing, None, like)
