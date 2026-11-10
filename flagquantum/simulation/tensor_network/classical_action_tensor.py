# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _tensor_from_classical.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_from_classical.py

"""Enumerate a classical register action as an explicit phase-free dense tensor."""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ..classical_action import ClassicalValueCallback
from .register_tensor import register_tensor_shape


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
                    bits = bits.reshape(
                        register.shape + (int(register.dtype.num_qubits),)
                    )
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
    except ValueError as error:
        raise ValueError(
            f"cannot compute tensor for {operation_name}: {error}"
        ) from error
    shape = (
        (1 << nright, 1 << nleft)
        if nleft and nright
        else ((1 << max(nleft, nright),) if nleft or nright else ())
    )
    return matrix.reshape(shape)
