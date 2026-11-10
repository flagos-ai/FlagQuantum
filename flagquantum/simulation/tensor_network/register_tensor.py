# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _tensor_data_manipulation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_data_manipulation.py

"""Dense register tensors in output-then-input, row-major element order."""

import math

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import RegisterSide
from ...core.ir.program.signatures import Signature

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
    return tuple(
        _word_dimension(register.bitsize)
        for ports in (signature.rights(), signature.lefts())
        for register in ports
        for _ in register.all_idxs()
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
    count = len(dimensions)
    indices: list[int | slice] = [slice(size) for size in dimensions * 2]
    position = 0
    for register in signature:
        for _ in register.all_idxs():
            if register.side == RegisterSide.LEFT:
                indices[position] = 0
            elif register.side == RegisterSide.RIGHT:
                indices[position + count] = 0
            position += 1
    result: DenseRegisterTensor = matrix.reshape(dimensions * 2)[tuple(indices)]
    assert result.shape == target_shape
    return result


def register_identity_tensor(signature: Signature) -> DenseRegisterTensor:
    """Build the source identity/projection; storage grows as 16 * 4**n bytes.

    n counts all declared bits, including directional ports. This explicit dense
    inspection path does not select a backend or replace circuit execution.
    """
    register_tensor_shape(signature)
    dimension = math.prod(
        _word_dimension(register.bitsize)
        for register in signature
        for _ in register.all_idxs()
    )
    return register_tensor_from_matrix(
        np.eye(dimension, dtype=np.complex128), signature
    )
