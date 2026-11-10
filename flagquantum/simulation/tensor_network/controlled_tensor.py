# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ControlledBase._tensor_data.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Insert a child tensor into the active block of a controlled identity."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ..value_dimensions import ArrayValue
from .register_tensor import register_identity_tensor, register_tensor_shape


def controlled_register_tensor(
    controls: Signature,
    child_signature: Signature,
    patterns: Sequence[ArrayValue],
    child_tensor: Callable[[], NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Build the pinned source's finite THRU-only dense controlled tensor.

    Control declarations precede child declarations; output axes precede input
    axes, with one axis per row-major word element. The child callback is called
    once, after identity allocation and before symbolic-pattern rejection.
    Raw integer pattern values index both control sides as in the source,
    including NumPy negative indexing. No signed-value codec is applied.

    This is an explicit exponential-memory inspection path, not a decomposition
    or a gate/resource estimator. Inputs remain unchanged. Invalid layout,
    symbolic patterns and incompatible child tensors raise ValueError; raw
    out-of-range indices retain NumPy's IndexError.
    """
    if not controls.thru_registers_only or not child_signature.thru_registers_only:
        raise ValueError("controlled dense tensors require THRU registers")
    if len(controls) != len(patterns):
        raise ValueError("one activation pattern is required per control register")
    signature = Signature((*controls, *child_signature))
    data = register_identity_tensor(signature)
    child_shape = register_tensor_shape(child_signature)
    child = child_tensor()
    if child_shape:
        child = child.reshape(child_shape)
    if is_symbolic_value(*patterns):
        raise ValueError("cannot compute active tensor space for symbolic patterns")
    indices: list[int | slice] = [slice(size) for size in data.shape]
    output_count = len(data.shape) // 2
    position = 0
    for register, pattern in zip(controls, patterns, strict=True):
        if not isinstance(pattern, np.ndarray) or pattern.shape != register.shape:
            raise ValueError("activation pattern shape must match its control register")
        for index in np.ndindex(pattern.shape):
            indices[position] = indices[position + output_count] = int(pattern[index])
            position += 1
    if not isinstance(data, np.ndarray):
        # The pinned scalar identity is a NumPy scalar, which cannot be assigned.
        raise TypeError("scalar controlled identity does not support item assignment")
    data[tuple(indices)] = child
    return data
