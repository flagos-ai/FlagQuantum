# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _tensor_data_manipulation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_data_manipulation.py

"""Dense register tensors in output-then-input, row-major element order."""

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .register_tensor_data import (
    project_register_tensor,
    register_identity_data,
    register_word_shape,
)

DenseRegisterTensor = NDArray[np.complex128] | np.complex128


def _word_dimension(width: object) -> int:
    if type(width) is not int or width < 0:
        raise ValueError("dense register tensors require finite nonnegative widths")
    return 1 << width


def register_tensor_shape(signature: Signature) -> tuple[int, ...]:
    """Return one dimension per word element, outputs before inputs.

    This numerical boundary requires finite nonnegative widths and dimensions.
    It interprets stored bit words without a dtype codec or physical allocation.
    """
    for register in signature:
        _word_dimension(register.bitsize)
        if any(type(size) is not int or size < 0 for size in register.shape):
            raise ValueError("dense register tensors require nonnegative array shapes")
    # Validation above proves raw word dimensions are concrete integers.
    return cast(tuple[int, ...], register_word_shape(signature))


def _projection_signature(signature: Signature) -> Signature:
    # The validated native boundary already admits a shared name on separate
    # LEFT/RIGHT ports. Names do not affect numerical axes; distinguish those
    # declarations before the raw source's THRU-signature reconstruction.
    if len({reg.name for reg in signature}) == len(signature):
        return signature
    return Signature(
        Register(str(i), reg.dtype, reg.shape, reg.side)
        for i, reg in enumerate(signature)
    )


def register_tensor_from_matrix(
    matrix: NDArray[np.complex128], signature: Signature
) -> DenseRegisterTensor:
    """Project a full word matrix onto zero at missing input/output ports.

    The matrix covers every declaration as if THRU. LEFT declarations select
    output zero and RIGHT declarations select input zero. This preserves the
    source projection rather than simulating free as a trace or a reset.
    """
    target_shape = register_tensor_shape(signature)
    dimensions = tuple(
        _word_dimension(register.bitsize)
        for register in signature
        for _ in register.all_idxs()
    )
    dimension = math.prod(dimensions)
    if matrix.shape != (dimension, dimension):
        raise ValueError("matrix shape does not match the complete register signature")
    result = cast(
        DenseRegisterTensor,
        project_register_tensor(matrix, _projection_signature(signature)),
    )
    assert result.shape == target_shape
    return result


def register_identity_tensor(signature: Signature) -> DenseRegisterTensor:
    """Build the source identity/projection; storage grows as 16 * 4**n bytes.

    n counts all declared bits, including directional ports. This explicit dense
    inspection path does not select a backend or replace circuit execution.
    """
    register_tensor_shape(signature)
    return cast(
        DenseRegisterTensor, register_identity_data(_projection_signature(signature))
    )
