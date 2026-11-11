"""Reconstruct numerical field metadata without changing executable type admission."""

from importlib import import_module

from ..core.ir.program.registers import RegisterType
from .finite_field_encoding import FiniteFieldEncoding


def _field_parameters(dtype: RegisterType) -> tuple[bool, dict[str, int]]:
    """Validate the existing projection's exact finite shape before optional imports."""
    classical = dtype.parameters[:1] == (("storage", "classical"),)
    entries = dtype.parameters[1:] if classical else dtype.parameters
    base = ("characteristic", "degree", "order")
    polynomial = (
        "polynomial",
        "polynomial_field_characteristic",
        "polynomial_field_degree",
        "polynomial_field_primitive_element",
        "polynomial_field_modulus",
    )
    names = tuple(name for name, _ in entries)
    if (
        dtype.encoding != "finite_field"
        or type(dtype.quantum_bits) is not int
        or type(dtype.classical_bits) is not int
        or names not in (base, base + polynomial)
        or any(type(value) is not int for _, value in entries)
    ):
        raise ValueError(
            "field codecs require concrete projected finite-field metadata"
        )
    parameters: dict[str, int] = {}
    for name, value in entries:
        assert type(value) is int
        parameters[name] = value
    characteristic, degree, order = (parameters[name] for name in base)
    if characteristic < 2 or degree < 1 or order != characteristic**degree:
        raise ValueError(
            "field order must equal characteristic raised to positive degree"
        )
    width = (order - 1).bit_length()
    if (dtype.quantum_bits, dtype.classical_bits) != (
        (0, width) if classical else (width, 0)
    ):
        raise ValueError("field storage widths differ from declared order")
    return classical, parameters


def resolve_finite_field_encoding(dtype: RegisterType) -> FiniteFieldEncoding:
    """Invert concrete quantum/classical field projection with an exact roundtrip.

    Polynomial coefficient-field identity and explicit None are retained. Numerical
    library validation remains lazy where the original codec was lazy. This is a
    numerical metadata adapter, not Core execution admission; odd and classical
    fields remain refused by the separate executable binary-field entry.
    """
    classical, parameters = _field_parameters(dtype)
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
    if encoding.register_type(classical=classical) != dtype:
        raise ValueError("resolved field encoding differs from declared metadata")
    return encoding
