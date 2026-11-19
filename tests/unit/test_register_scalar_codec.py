"""Compare shared scalar register encoding with each unchanged source codec."""

import pytest

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.simulation.classical_register_codec import encode_register_value

try:
    from tests.unit.test_classical_register_codec import KINDS, pair
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


@pytest.mark.parametrize("kind", KINDS)
def test_each_stored_scalar_matches_original(kind: str) -> None:
    native, source = pair(kind)
    for value in (range(4) if kind == "opaque" else source.get_classical_domain()):
        assert encode_register_value(Register("q", native), value) == source.to_bits(
            value
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", [-7, 9])
def test_original_scalar_domain_failure_is_retained(kind: str, value: int) -> None:
    native, source = pair(kind)
    try:
        expected = source.to_bits(value)
    except (ValueError, AssertionError, TypeError) as error:
        with pytest.raises(type(error)):
            encode_register_value(Register("q", native), value)
    else:
        assert encode_register_value(Register("q", native), value) == expected


def test_unrecognized_encoding_fails_at_numerical_owner() -> None:
    with pytest.raises(ValueError, match="scalar bit codec"):
        encode_register_value(Register("q", RegisterType("unknown", 2)), 0)


@pytest.mark.parametrize("polynomial", [False, True])
@pytest.mark.parametrize("classical", [False, True])
def test_field_and_polynomial_scalar_composition(
    polynomial: bool, classical: bool
) -> None:
    from qualtran import CGF, QGF, CGFPoly, QGFPoly

    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from flagquantum.simulation.finite_field_polynomial_encoding import (
        FiniteFieldPolynomialEncoding,
    )

    field = FiniteFieldEncoding(2, 2)
    encoding = FiniteFieldPolynomialEncoding(1, field) if polynomial else field
    source = (
        (CGFPoly if classical else QGFPoly)(1, QGF(2, 2))
        if polynomial
        else (CGF if classical else QGF)(2, 2)
    )
    register = Register("q", encoding.register_type(classical=classical))
    for value in source.get_classical_domain():
        assert encode_register_value(register, value) == source.to_bits(value)
