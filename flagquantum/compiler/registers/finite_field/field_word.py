"""Validate concrete binary-field words for field-specific stored components."""

from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from .field_multiplication import _validate_field


def binary_field_word_width(dtype: RegisterType) -> int:
    """Separate binary arithmetic admission from generic raw field storage."""
    if dtype.encoding != "finite_field":
        raise ValueError("Field operations require a binary field descriptor")
    width = executable_register_width(dtype)
    if dict(dtype.parameters).get("characteristic") != 2:
        raise ValueError(f"unsupported executable register descriptor {dtype!r}")
    return width


def field_word_width(dtype: RegisterType, modulus: int) -> int:
    """Keep canonical descriptor identity and the explicit binary polynomial."""
    width = binary_field_word_width(dtype)
    _validate_field(width, modulus)
    if dict(dtype.parameters).get("polynomial", modulus) != modulus:
        raise ValueError("Field modulus must match the field descriptor")
    return width


def raw_field_word_width(dtype: RegisterType) -> int:
    """Validate existing quantum field storage without numerical arithmetic admission."""
    if dtype.encoding != "finite_field" or dtype.classical_bits != 0:
        raise ValueError("Raw field components require a quantum field descriptor")
    return executable_register_width(dtype)
