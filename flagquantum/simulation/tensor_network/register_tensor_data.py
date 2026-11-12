# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 simulation/tensor/_tensor_data_manipulation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_data_manipulation.py

"""Raw word-axis shapes, control slices and directional tensor projections."""

from collections.abc import Callable, Sequence
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal
from ...core.value_parameters import is_symbolic_value
from ..value_dimensions import ArrayValue


def register_word_shapes(
    signature: Signature,
) -> tuple[tuple[SymbolicReal, ...], tuple[SymbolicReal, ...]]:
    """Return output and input word dimensions without finite-width validation.

    Evaluate inputs first, matching the source's failure order. Symbolic bit
    widths remain symbolic powers; symbolic array shapes retain metadata errors.
    """
    inputs = tuple(2**reg.bitsize for reg in signature.lefts() for _ in reg.all_idxs())
    outputs = tuple(
        2**reg.bitsize for reg in signature.rights() for _ in reg.all_idxs()
    )
    return outputs, inputs


def register_word_shape(signature: Signature) -> tuple[SymbolicReal, ...]:
    """Concatenate raw output and input dimensions in source order."""
    outputs, inputs = register_word_shapes(signature)
    return outputs + inputs


def register_active_indices(
    signature: Signature, patterns: Sequence[ArrayValue]
) -> tuple[int | slice, ...]:
    """Index both sides of the leading control words with raw integer patterns.

    This raw boundary assumes controls precede child registers. It deliberately
    retains NumPy indexing, including negatives, rather than applying a codec.
    Validated callers enforce pattern count, shape and register layout.
    """
    if is_symbolic_value(
        *(reg.dtype for reg in islice(signature, len(patterns)))
    ) or is_symbolic_value(*patterns):
        raise ValueError("cannot compute active tensor space for symbolic patterns")
    outputs, inputs = register_word_shapes(signature)
    indices: list[int | slice] = [slice(size) for size in outputs + inputs]
    position = 0
    for pattern in patterns:
        assert isinstance(pattern, np.ndarray)
        for index in np.ndindex(pattern.shape):
            indices[position] = int(pattern[index])
            indices[position + len(outputs)] = int(pattern[index])
            position += 1
    return tuple(indices)


def _declared_bits(signature: Signature) -> SymbolicInt:
    return sum(reg.total_bits() for reg in signature)


def project_register_tensor(
    matrix: NDArray[Any], signature: Signature
) -> NDArray[Any] | np.generic:
    """Reshape all declared words as THRU, then project absent ports onto zero.

    LEFT words select output zero; RIGHT words select input zero. This is not
    a trace, reset or unitarity check. Original shape assertions and NumPy
    dtype, subclass, scalar-result and failure behavior remain observable.
    """
    assert matrix.shape == (2 ** _declared_bits(signature),) * 2
    thru = Signature(
        Register(reg.name, reg.dtype, reg.shape, RegisterSide.THRU) for reg in signature
    )
    shape = register_word_shape(thru)
    # NumPy owns rejection of raw symbolic/nonintegral dimensions at this
    # external boundary. Do not coerce them or preempt its original errors.
    reshape: Callable[..., NDArray[Any]] = matrix.reshape
    data = reshape(shape)
    result: NDArray[Any] | np.generic = data[_zero_projection_indices(signature, shape)]
    assert result.shape == register_word_shape(signature)
    return result


def _zero_projection_indices(
    signature: Signature, shape: tuple[SymbolicReal, ...]
) -> tuple[int | slice, ...]:
    """Share absent-port zero slices between NumPy and PyTorch inspection."""
    count = len(shape) // 2
    indices: list[int | slice] = [slice(size) for size in shape]
    position = 0
    for reg in signature:
        if reg.side == RegisterSide.LEFT:
            for _ in reg.all_idxs():
                indices[position] = 0
                position += 1
        if reg.side == RegisterSide.RIGHT:
            for _ in reg.all_idxs():
                indices[position + count] = 0
                position += 1
        if reg.side == RegisterSide.THRU:
            position += int(np.prod(reg.shape))
    return tuple(indices)


def register_identity_data(signature: Signature) -> NDArray[Any] | np.generic:
    """Allocate the source complex128 identity and apply directional projection.

    Storage is exponential in all declared bits. This raw inspection boundary
    leaves dimensions and allocation failures to NumPy, as in the source.
    """
    make_identity: Callable[..., NDArray[np.complex128]] = np.eye
    return project_register_tensor(
        make_identity(2 ** _declared_bits(signature), dtype=np.complex128), signature
    )
