# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 GF2PolyAdd/GF2PolyAddK scalar callbacks.

"""Polynomial arithmetic callbacks at the optional numerical field boundary."""

from typing import Any

from ..core.value_parameters import is_symbolic_value
from .finite_field_polynomial_encoding import FiniteFieldPolynomialEncoding


def validate_polynomial_constant(
    encoding: FiniteFieldPolynomialEncoding, constant: Any
) -> None:
    """Retain the original degree-first and nominal field-identity checks.

    Symbolic degree skips both checks. Concrete degree with symbolic field
    skips only field identity, matching the original constructor validator.
    Executable coefficient tuples are validated separately by Compiler.
    """
    if not is_symbolic_value(encoding.degree):
        if constant.degree > encoding.degree:
            raise ValueError(
                f"Degree of constant polynomial must be <= {encoding.degree}"
            )
        if not is_symbolic_value(encoding.field.register_type()):
            if constant.field is not encoding.field.gf_type:
                raise ValueError(
                    "Constant polynomial must be defined over galois field "
                    f"{encoding.field.gf_type}"
                )


def polynomial_add_classical_values(source: Any, target: Any) -> tuple[Any, Any]:
    """Preserve source identity and the numerical polynomial addition behavior."""
    return source, source + target


def polynomial_add_constant_classical_value(target: Any, constant: Any) -> Any:
    """Execute the original callback without adding constructor validation."""
    return target + constant
