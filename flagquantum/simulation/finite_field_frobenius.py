# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Square numerical methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_square.py

"""Numerical Frobenius matrices and source-compatible finite-field values."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .finite_field_encoding import FiniteFieldEncoding


def field_frobenius_matrix(
    encoding: FiniteFieldEncoding, count: int = 1
) -> NDArray[np.int64]:
    """Construct the mathematical x -> x**(2**count) binary coefficient matrix.

    Columns and rows run from least to most significant polynomial coefficient.
    Count zero is the identity. Periodicity bounds numerical exponent size.
    The frozen source's k=0 and odd-power matrix discrepancies are documented
    separately from this mathematical matrix.
    """
    width = encoding.degree
    if encoding.characteristic != 2 or type(width) is not int or width < 1:
        raise ValueError("Frobenius matrices require a concrete positive binary field")
    if type(count) is not int or count < 0:
        raise ValueError("Frobenius count must be a nonnegative concrete integer")
    field = encoding.gf_type
    columns = [int(field(1 << i) ** (2 ** (count % width))) for i in range(width)]
    return np.array(
        [[(column >> row) & 1 for column in columns] for row in range(width)],
        dtype=np.int64,
    )


def field_frobenius_classical_value(
    encoding: FiniteFieldEncoding,
    value: Any,
    count: int = 1,
    *,
    adjoint: bool = False,
) -> Any:
    """Retain the original scalar-class assertion and repeated square-root inverse.

    Field arrays and optional galois result types stay inside this numerical
    boundary. Parameter failure behavior follows the original value callback.
    Concrete executable wire admission is owned by Compiler.
    """
    assert isinstance(value, encoding.gf_type)
    if adjoint:
        for _ in range(count):
            value = np.sqrt(value)
        return value
    return value ** (2**count)
