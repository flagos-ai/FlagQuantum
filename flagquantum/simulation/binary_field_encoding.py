"""Resolve admitted binary-field metadata at the Simulation numerical boundary.

Core keeps vendor-neutral declarations. This inverse of the existing field
projection reconstructs numerical objects only on demand; it does not expand
executable type admission or select a backend.
"""

from ..core.ir.program.register_verifier import executable_register_width
from ..core.ir.program.registers import RegisterType
from .finite_field_encoding import FiniteFieldEncoding
from .finite_field_metadata import resolve_finite_field_encoding


def resolve_binary_field_encoding(dtype: RegisterType) -> FiniteFieldEncoding:
    """Reconstruct a field declaration without discarding polynomial identity.

    Only existing executable binary quantum descriptors are admitted. Absent
    polynomial metadata means explicit None, not an omitted/default polynomial.
    Coefficient-field characteristic, degree, primitive element and modulus are
    retained, even for target degree one where the original numerical codec
    ignores the polynomial. A roundtrip verifies the complete declaration.

    Optional galois imports occur after Core admission. The target numerical
    field remains lazy; callers use its existing scalar/array codecs. Invalid
    numerical-library parameters propagate without repair or integer coercion.
    """
    executable_register_width(dtype)
    if dtype.encoding != "finite_field":
        raise ValueError("binary field resolution requires finite-field metadata")
    return resolve_finite_field_encoding(dtype)
