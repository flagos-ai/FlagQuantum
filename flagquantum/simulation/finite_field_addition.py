# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Addition/GF2AddK.on_classical_vals and
# gf_utils.qgf_converter at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Numerical field construction and addition at the optional galois boundary."""

from importlib import import_module
from typing import Any

from ..core.value_parameters import is_symbolic_value
from .finite_field_encoding import FiniteFieldEncoding


def binary_field_encoding(value: Any) -> FiniteFieldEncoding:
    """Resolve a native encoding, degree, polynomial or descending powers.

    Dynamic galois polynomial classes stop at this Simulation boundary. Existing
    native encodings preserve identity. Symbolic degrees remain lazy, with the
    original conversion dispatch and polynomial construction order.
    """
    if isinstance(value, FiniteFieldEncoding):
        return value
    if isinstance(value, int) or is_symbolic_value(value):
        return FiniteFieldEncoding(2, value)
    polynomial_type = import_module("galois").Poly
    polynomial = (
        value if isinstance(value, polynomial_type) else polynomial_type.Degrees(value)
    )
    return FiniteFieldEncoding(2, polynomial.degree, polynomial)


def field_add_classical_values(
    encoding: FiniteFieldEncoding, source: Any, target: Any
) -> tuple[Any, Any]:
    """Preserve the field-typed source and add it into the field-typed target.

    Assertions retain the original scalar-class check, including its rejection
    of ordinary Python integers and scalars from a different field.
    """
    assert isinstance(source, encoding.gf_type) and isinstance(target, encoding.gf_type)
    return source, source + target


def field_add_constant_classical_value(
    encoding: FiniteFieldEncoding, target: Any, constant: Any
) -> Any:
    """Use the field constructor's original constant validation before addition."""
    assert isinstance(target, encoding.gf_type)
    return target + encoding.gf_type(constant)
