"""Resolve admitted binary-field metadata at the Simulation numerical boundary.

Core keeps vendor-neutral declarations. This inverse of the existing field
projection reconstructs numerical objects only on demand; it does not expand
executable type admission or select a backend.
"""

from importlib import import_module

from ..core.ir.program.register_verifier import executable_register_width
from ..core.ir.program.registers import RegisterType
from .finite_field_encoding import FiniteFieldEncoding


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
    parameters: dict[str, int] = {}
    for name, value in dtype.parameters:
        # The authoritative Core admission above requires exact Python ints.
        assert type(value) is int
        parameters[name] = value
    characteristic, degree = parameters["characteristic"], parameters["degree"]
    if "polynomial" not in parameters:
        encoding = FiniteFieldEncoding(characteristic, degree, None)
    else:
        galois = import_module("galois")
        coefficient_degree = parameters["polynomial_field_degree"]
        coefficient_field = galois.GF(
            parameters["polynomial_field_characteristic"],
            coefficient_degree,
            irreducible_poly=(
                parameters["polynomial_field_modulus"]
                if coefficient_degree > 1
                else None
            ),
            primitive_element=parameters["polynomial_field_primitive_element"],
            verify=False,
            compile="python-calculate",
        )
        polynomial = galois.Poly.Int(parameters["polynomial"], field=coefficient_field)
        encoding = FiniteFieldEncoding(characteristic, degree, polynomial)
    if encoding.register_type() != dtype:
        raise ValueError("resolved field encoding differs from declared metadata")
    return encoding
