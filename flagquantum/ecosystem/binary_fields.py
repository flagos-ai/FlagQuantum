# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/circuit/subs/gf_arithmetic.pybind.cc.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Convert external binary field declarations into neutral owned scalar values."""

from typing import SupportsIndex, SupportsInt

from ..core._binary_field import BinaryField
from ..core._binary_polynomial import parse


def _characteristic(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 2:
        raise ValueError("Binary field objects require integer characteristic two")


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise ValueError("Binary polynomial requires an integer-index value")
    result = int(value)
    if result < 0 or result.bit_length() > 1024:
        raise ValueError("Binary polynomial must fit 1024 nonnegative coefficient bits")
    return result


def binary_polynomial_value(value: object) -> int:
    """Consume a polynomial string, index value or characteristic-two polynomial."""
    if isinstance(value, str):
        return parse(value)
    if all(hasattr(value, key) for key in ("field", "degree", "coeffs")):
        _characteristic(getattr(getattr(value, "field"), "characteristic"))
        # Original galois.Poly conversion uses int rather than __index__.
        if not isinstance(value, SupportsInt):
            raise ValueError("External polynomial requires integer conversion")
        result = int(value)
        return _integer(result)
    if hasattr(type(value), "characteristic"):
        _characteristic(getattr(type(value), "characteristic"))
    return _integer(value)


def _external_field(value: object) -> bool:
    return all(
        hasattr(value, key) for key in ("characteristic", "degree", "irreducible_poly")
    )


def binary_field_value(
    degree: object = None,
    modulus: object = None,
    primitive_element: object = None,
) -> BinaryField:
    """Retain the source constructor dispatch at the external-object boundary."""
    if _external_field(degree):
        if modulus is not None:
            raise ValueError("Explicit modulus cannot accompany an external field")
        _characteristic(getattr(degree, "characteristic"))
        modulus = getattr(degree, "irreducible_poly")
        if primitive_element is None:
            primitive_element = getattr(degree, "primitive_element", None)
        degree = getattr(degree, "degree")
    if degree is not None:
        if not isinstance(degree, int) or isinstance(degree, bool):
            raise ValueError("Binary field degree must be an integer or None")
        degree = int(degree)
    if degree is not None and not 1 <= degree <= 512:
        raise ValueError("Binary field degree must be between 1 and 512")
    return BinaryField(
        degree,
        None if modulus is None else binary_polynomial_value(modulus),
        (
            None
            if primitive_element is None
            else binary_polynomial_value(primitive_element)
        ),
    )


def resolve_binary_field(value: object, expected_degree: int) -> BinaryField:
    """Resolve only owned values, external field declarations or source defaults."""
    if value is None:
        field = binary_field_value(expected_degree)
    elif isinstance(value, BinaryField):
        field = value
    elif _external_field(value):
        field = binary_field_value(value)
    else:
        raise ValueError("Field requires an owned value, external field or None")
    if field.degree != expected_degree:
        raise ValueError("Field degree must match the register size")
    return field
