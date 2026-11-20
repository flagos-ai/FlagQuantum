"""Validate concrete binary-field words for field-specific stored components."""

from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from .field_multiplication import _validate_field


def field_word_width(dtype: RegisterType, modulus: int) -> int:
    """Keep canonical descriptor identity and the explicit binary polynomial."""
    if dtype.encoding != "finite_field":
        raise ValueError("Field operations require a binary field descriptor")
    width = executable_register_width(dtype)
    _validate_field(width, modulus)
    if dict(dtype.parameters).get("polynomial", modulus) != modulus:
        raise ValueError("Field modulus must match the field descriptor")
    return width
