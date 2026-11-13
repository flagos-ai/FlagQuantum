# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 gf2_multiplication numerical callbacks.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Field products and binary coefficient callbacks at the numerical boundary."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.value_parameters import is_symbolic_value
from .finite_field_encoding import FiniteFieldEncoding


def field_product_classical_values(
    encoding: FiniteFieldEncoding,
    left: Any,
    right: Any,
    result: Any = None,
    *,
    plus_equal_prod: bool = False,
) -> tuple[Any, Any, Any]:
    """Retain typed factors and the original product/plus-equal scalar action."""
    field = encoding.gf_type
    assert isinstance(left, field) and isinstance(right, field)
    if plus_equal_prod or result is not None:
        assert isinstance(result, field)
    target = result if plus_equal_prod else field(0)
    return left, right, target + left * right


def field_scale_classical_value(
    encoding: FiniteFieldEncoding, value: Any, constant: Any
) -> Any:
    """Use the original constant field construction and scalar multiplication."""
    return value * encoding.gf_type(constant)


def field_shift_classical_value(
    encoding: FiniteFieldEncoding, value: Any, count: Any, *, inverse: bool = False
) -> Any:
    """Retain the source's symbolic rejection and degree-one identity special case."""
    if is_symbolic_value(count):
        raise TypeError("Classical monomial action requires a concrete exponent")
    assert isinstance(value, encoding.gf_type)
    if count == 0 or encoding.degree == 1:
        return value
    return value * encoding.gf_type(2) ** (-count if inverse else count)


def field_scale_matrix(
    encoding: FiniteFieldEncoding, constant: Any
) -> NDArray[np.int64]:
    """Return least-significant-first columns of multiplication by a constant.

    The same construction supplies the MBU reduction matrix with constant x**m
    reduced into the field. Matrix construction does not synthesize gates.
    """
    width = int(encoding.bitsize)
    field = encoding.gf_type
    columns = [int(field(constant) * field(1 << i)) for i in range(width)]
    return np.array(
        [[(column >> row) & 1 for column in columns] for row in range(width)],
        dtype=np.int64,
    )


def binary_linear_classical_value(
    matrix: NDArray[np.integer[Any]],
    value: NDArray[np.integer[Any]],
    *,
    adjoint: bool = False,
) -> NDArray[np.int64]:
    """Apply the source numerical matrix or its true finite-field inverse.

    The original gate decomposition ignores its adjoint flag. This callback
    follows the independent classical method, which does invert the matrix.
    """
    from galois import GF

    binary = GF(2)
    assert isinstance(value, np.ndarray)
    transform = binary(matrix.astype(int))
    if adjoint:
        transform = binary(np.linalg.inv(transform))
    return np.asarray(transform @ binary(value), dtype=np.int64)


def binary_polynomial_product_classical_values(
    left: NDArray[np.integer[Any]],
    right: NDArray[np.integer[Any]],
    target: NDArray[np.integer[Any]],
    *,
    shift: Any = None,
) -> tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]], list[int]]:
    """Accumulate a carryless product in little-endian coefficient arrays.

    An optional positive shift multiplies the product by 1+x**shift. The
    original callback returns the target as a list and preserves input arrays.
    """
    from galois import Poly

    if is_symbolic_value(shift):
        raise TypeError("Classical polynomial action requires a concrete shift")
    assert isinstance(left, np.ndarray)
    assert isinstance(right, np.ndarray)
    assert isinstance(target, np.ndarray)
    product = Poly(left[::-1]) * Poly(right[::-1])
    result = Poly(target[::-1])
    if shift is None:
        result += product
    if shift is not None and shift > 0:
        result += product * Poly.Degrees([0, shift])
    coefficients = result.coefficients().tolist()
    return left, right, ([0] * (len(target) - len(coefficients)) + coefficients)[::-1]
