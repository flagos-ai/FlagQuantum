# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 GF2Inverse.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_inverse.py

"""Original field-inverse values and retained intermediate output words."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .finite_field_encoding import FiniteFieldEncoding


def field_inverse_classical_values(
    encoding: FiniteFieldEncoding, value: Any
) -> tuple[Any, Any, NDArray[Any]]:
    """Preserve the source field scalar, inverse and original junk array semantics.

    Zero maps to zero. Junk contains live intermediate values and uses the
    source's NumPy array conversion, including its empty-array dtype at degree
    one and two. Executable workspace cleanup is owned by Compiler.
    """
    assert isinstance(value, encoding.gf_type)
    width = encoding.degree
    hamming = int(width - 1).bit_count()
    top = int(width - 1).bit_length() - 1
    last = max(top + hamming - 1, top + 1)
    values = [value] + [0] * last
    for i in range(1, top + 1):
        values[i] = values[i - 1] ** (2 ** (2 ** (i - 1)) + 1)
    bits = [-1] + [
        top - i for i, bit in enumerate(np.binary_repr(width - 1)) if bit == "1"
    ]
    for i in range(1, hamming):
        values[top + i - 1] = values[top + i - 1] ** (2 ** (2 ** bits[i + 1]))
        values[top + i] = values[top + i - 1] * values[bits[i + 1]]
    if hamming == 1:
        values[top], values[last] = values[last], values[top]
    values[last] = values[last] ** 2
    result = value ** (-1) if value else encoding.gf_type(0)
    return value, result, np.array(values[1:-1])
