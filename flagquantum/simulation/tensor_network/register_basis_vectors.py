# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _IntVector decomposition and _ZVector.my_tensors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Supply binary computational-basis kets and covectors on existing labels."""

from collections.abc import Mapping

import numpy as np
import torch

from ...core.ir.program.signatures import Signature
from ..bit_encoding import bit_to_bits
from ..classical_basis_vectors import integer_basis_classical_values
from ..classical_bitwise import IntegerValue
from ..integer_encoding import integer_to_bits
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups


def register_basis_vector_nodes(
    signature: Signature,
    connections: Mapping[str, tuple[tuple[int, ...], ...]],
    values: Mapping[str, IntegerValue] | None,
    like: torch.Tensor,
    *,
    state: bool = True,
) -> tuple[TensorNetworkNode, ...]:
    """Emit one rank-one tensor per bit, in signature/row-major/MSB order.

    Ports must be exclusively RIGHT for kets or LEFT for covectors, and quantum.
    Values are unsigned integer basis indices, independently of port encoding,
    matching IntState/IntEffect rather than dtype-specific value conversion.
    Shaped values use the source asarray/index traversal without broadcasting.
    None selects all-zero words for the existing zero-state supplier.
    Existing vector validation, scalar encoders and label binding own failures.
    Missing values and indices propagate; all labels must be distinct.

    Data follows like's dtype/device. No physical preparation, measurement,
    reset or normalization occurs. Empty arrays and zero-width words emit no
    nodes, retaining the native zero-state boundary; the original zero-width
    IntVector decomposition can fail at its join. Empty contraction remains the
    caller's responsibility. Node storage is linear in the number of live bits.
    """
    if type(state) is not bool:
        raise ValueError("basis-vector direction requires a built-in boolean")
    if tuple(signature.lefts() if state else signature.rights()) or any(
        r.dtype.classical_bits for r in signature
    ):
        raise ValueError("basis-vector tensors require one-sided quantum ports")
    groups = register_tensor_label_groups(
        signature, {} if state else connections, connections if state else {}
    )
    labels = tuple(label for group in groups for label in group)
    if len(labels) != len(set(labels)):
        raise ValueError("basis-vector bit labels must be distinct")
    nodes: list[TensorNetworkNode] = []
    for register in signature:
        width = register.bitsize
        assert isinstance(width, int)
        if not width or any(size == 0 for size in register.shape):
            continue
        data = (
            np.asarray(values[register.name])
            if values is not None and register.shape
            else None
        )
        for position, index in enumerate(register.all_idxs()):
            value = (
                data[index]
                if data is not None
                else (0 if values is None else values[register.name])
            )
            integer_basis_classical_values(value, width)
            bits = bit_to_bits(value) if width == 1 else integer_to_bits(value, width)
            for bit, label in zip(
                bits, connections[register.name][position], strict=True
            ):
                nodes.append(
                    TensorNetworkNode(
                        torch.tensor(
                            [1 - bit, bit], dtype=like.dtype, device=like.device
                        ),
                        (label,),
                        name=(
                            "register_basis_state" if state else "register_basis_effect"
                        ),
                    )
                )
    return tuple(nodes)
