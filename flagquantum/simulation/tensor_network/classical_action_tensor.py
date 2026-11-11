# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _tensor_from_classical.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_from_classical.py

"""Enumerate a classical register action as an explicit phase-free dense tensor."""

from collections.abc import Callable, Mapping
from itertools import product
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ..classical_action import ClassicalValueCallback
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups
from .register_tensor import register_tensor_shape
from .register_tensor_binding import bind_register_tensor


def _classical_action_tensor_data(
    signature: Signature,
    action: ClassicalValueCallback[tuple[object, ...]],
    decode_input: Callable[[Register, NDArray[Any]], object],
    encode_output: Callable[[Register, NDArray[Any]], NDArray[np.uint8]],
) -> NDArray[np.float64]:
    """Enumerate binary axes without the dense entry point's error wrapping."""
    register_tensor_shape(signature)
    left = tuple(signature.lefts())
    right = tuple(signature.rights())
    counts = tuple(int(register.total_bits()) for register in left)
    nleft = sum(counts)
    nright = sum(int(register.total_bits()) for register in right)
    if nleft + nright > 40:
        raise ValueError(f"tensor is too large: {nleft + nright} total qubits")
    matrix = np.zeros((2,) * (nleft + nright), dtype=np.float64)
    splits = np.cumsum(counts)
    for input_bits in product((0, 1), repeat=nleft):
        *words, last = np.split(input_bits, splits)
        assert np.size(last) == 0
        values: dict[str, object] = {}
        for register, bits in zip(left, words, strict=True):
            if register.shape:
                bits = bits.reshape(register.shape + (int(register.dtype.num_qubits),))
            values[register.name] = decode_input(register, bits)
        outputs = action(**values)
        # Preserve source zip truncation and output truthiness semantics.
        output_bits = (
            np.concatenate(
                [
                    encode_output(register, np.asarray(value)).flatten()
                    for register, value in zip(right, outputs, strict=False)
                ]
            )
            if outputs
            else np.array([])
        )
        matrix[tuple([*np.atleast_1d(output_bits), *input_bits])] = 1
    return matrix


def classical_action_tensor(
    signature: Signature,
    action: ClassicalValueCallback[tuple[object, ...]],
    decode_input: Callable[[Register, NDArray[Any]], object],
    encode_output: Callable[[Register, NDArray[Any]], NDArray[np.uint8]],
    *,
    operation_name: str,
) -> NDArray[np.float64]:
    """Enumerate input bits, call a value action, and mark output/input entries.

    Input and output ports follow signature order, row-major words and MSB-first
    bits. The action receives named decoded inputs and returns output values in
    right-register order. Scalar input bit arrays are flat; array inputs have
    shape `register.shape + (register.dtype.num_qubits,)`, matching the pinned
    source even for classical ports. Caller codecs own value interpretation.
    Output codecs receive np.asarray(value) and return arrays of encoded bits.

    Values are zero or one, with no inferred phase, normalization or unitarity.
    Non-injective actions are permitted. Dense storage is 8 * 2**(left+right)
    bytes; the original 40-total-bit ceiling is a refusal limit, not a memory
    guarantee. Finite metadata is required before allocation. ValueError is
    wrapped with the operation name; other codec/action errors propagate.
    Use the existing register tensor binder to attach the result to nodes.
    """
    try:
        matrix = _classical_action_tensor_data(
            signature, action, decode_input, encode_output
        )
    except ValueError as error:
        raise ValueError(
            f"cannot compute tensor for {operation_name}: {error}"
        ) from error
    nleft = sum(int(register.total_bits()) for register in signature.lefts())
    nright = sum(int(register.total_bits()) for register in signature.rights())
    shape = (
        (1 << nright, 1 << nleft)
        if nleft and nright
        else ((1 << max(nleft, nright),) if nleft or nright else ())
    )
    return matrix.reshape(shape)


def classical_action_tensor_node(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    action: ClassicalValueCallback[tuple[object, ...]],
    decode_input: Callable[[Register, NDArray[Any]], object],
    encode_output: Callable[[Register, NDArray[Any]], NDArray[np.uint8]],
    like: torch.Tensor,
    *,
    operation_name: str,
) -> TensorNetworkNode:
    """Attach a phase-free action table to an existing pure-state leaf boundary.

    This is the native equivalent of source my_tensors_from_classical_action:
    the existing binder owns output-before-input binary axes and integer edges.
    The caller supplies known phase-free actions and codecs; nothing discovers
    or substitutes actions from a FunctionRecord name. Data uses like's dtype
    and device. The NumPy callback itself is not trainable.

    Classical ports are refused before callbacks, since the original tensor
    supplier indexes quantum bits while its table counts all bits. Invalid
    connections also fail before enumeration. These are explicit native refusal
    boundaries. Scalar quantum nodes are supported with explicit scalar data.
    Like the source leaf supplier, enumeration errors propagate without the
    dense entry point's operation-context wrapping.
    """
    if any(register.dtype.classical_bits for register in signature):
        raise ValueError("pure-state classical tensor suppliers require quantum ports")
    register_tensor_label_groups(signature, incoming, outgoing)
    table = _classical_action_tensor_data(
        signature, action, decode_input, encode_output
    )
    return bind_register_tensor(
        torch.as_tensor(table, dtype=like.dtype, device=like.device),
        signature,
        incoming,
        outgoing,
        name=operation_name,
    )
