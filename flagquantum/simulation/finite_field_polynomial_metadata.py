"""Reconstruct numerical polynomial codecs through the existing field projection."""

from ..core.ir.program.registers import RegisterType
from .finite_field_metadata import resolve_finite_field_encoding
from .finite_field_polynomial_encoding import FiniteFieldPolynomialEncoding


def resolve_finite_field_polynomial_encoding(
    dtype: RegisterType,
) -> FiniteFieldPolynomialEncoding:
    """Invert a concrete polynomial projection without widening Core admission.

    The first numerical parameter is polynomial degree; all remaining parameters
    are the original coefficient-field projection. The field owner validates and
    restores that projection. No field arithmetic or polynomial normalization is
    repeated here. Scalar and array value semantics stay with the codec owner.
    """
    classical = dtype.parameters[:1] == (("storage", "classical"),)
    entries = dtype.parameters[1:] if classical else dtype.parameters
    if (
        dtype.encoding != "finite_field_polynomial"
        or type(dtype.quantum_bits) is not int
        or type(dtype.classical_bits) is not int
        or not entries
        or entries[0][0] != "polynomial_degree"
        or type(entries[0][1]) is not int
        or entries[0][1] < 0
    ):
        raise ValueError("polynomial codecs require concrete projected metadata")
    degree = entries[0][1]
    width = dtype.classical_bits if classical else dtype.quantum_bits
    other_width = dtype.quantum_bits if classical else dtype.classical_bits
    if other_width or width < 0 or width % (degree + 1):
        raise ValueError("polynomial storage widths differ from declared degree")
    field_type = RegisterType(
        "finite_field", width // (degree + 1), parameters=entries[1:]
    )
    field = resolve_finite_field_encoding(field_type)
    return FiniteFieldPolynomialEncoding(degree, field)
