"""Pinned finite-field construction, lazy properties and numerical codecs."""

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)
FIELDS = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (5, 1)]
RAW = FIELDS + [
    (4, 2),
    (1, 2),
    (2, 0),
    (2, -1),
    (0, 2),
    (True, 2),
    (2, True),
    (2.0, 2),
    (2, 2.0),
    (np.int64(2), np.int64(3)),
    (sympy.Integer(2), sympy.Integer(3)),
    (N, 2),
    (2, N),
    (N, N),
    (None, 2),
    (2, None),
    ("2", 2),
]


def source_factory(characteristic, degree, polynomial=OMITTED):
    from qualtran._infra.data_types import _GF

    if polynomial is OMITTED:
        return _GF(characteristic, degree)
    return _GF(characteristic, degree, polynomial)


def native_factory(characteristic, degree, polynomial=OMITTED):
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    if polynomial is OMITTED:
        return FiniteFieldEncoding(characteristic, degree)
    return FiniteFieldEncoding(characteristic, degree, polynomial)


@pytest.fixture(params=["source", "native"])
def factory(request):
    return source_factory if request.param == "source" else native_factory


def outcome(call):
    try:
        value = call()
    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        RuntimeError,
    ) as error:
        message = str(error).replace("self.bitsize=", "width=")
        return type(error), message
    if isinstance(value, np.ndarray):
        return type(value), value.dtype, value.shape, value.tolist()
    return type(value), value


@pytest.mark.parametrize("characteristic,degree", RAW)
@pytest.mark.parametrize("polynomial", [OMITTED, None, [2, 1, 0], [3, 1, 0]])
def test_constructor_and_lazy_failures(factory, characteristic, degree, polynomial):
    def observe(construct):
        try:
            field = construct(characteristic, degree, polynomial)
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
            RuntimeError,
        ) as error:
            return "constructor", type(error), str(error)
        assert field.characteristic is characteristic
        assert field.degree is degree
        return (
            outcome(lambda: field.irreducible_poly),
            outcome(lambda: field.order),
            outcome(lambda: field.bitsize),
            outcome(lambda: field.gf_type),
        )

    assert observe(factory) == observe(source_factory)


@pytest.mark.parametrize("characteristic,degree", FIELDS)
def test_domain_scalar_round_trip_and_array_paths(factory, characteristic, degree):
    source = source_factory(characteristic, degree)
    field = factory(characteristic, degree)
    assert field.gf_type is source.gf_type
    assert list(field.get_domain()) == list(source.get_domain())
    for value in source.get_domain():
        assert field.to_bits(value) == source.to_bits(value)
        assert outcome(lambda: field.from_bits(source.to_bits(value))) == outcome(
            lambda: source.from_bits(source.to_bits(value))
        )
    bits = np.array(
        [source.to_bits(value) for value in source.get_domain()], dtype=np.uint8
    )
    assert outcome(lambda: field.from_bits_array(bits)) == outcome(
        lambda: source.from_bits_array(bits)
    )
    values = source.gf_type.elements
    # The default vectorizer extracts ordinary scalars before field validation.
    assert outcome(lambda: field.to_bits_array(values)) == outcome(
        lambda: source.to_bits_array(values)
    )
    assert field.gf_type is field.gf_type
    assert field.order is field.order
    assert field.bitsize is field.bitsize


@pytest.mark.parametrize("characteristic,degree", FIELDS)
@pytest.mark.parametrize("value", [0, True, -1, 1.0, np.int64(1), None, [1]])
def test_scalar_nominal_validation(factory, characteristic, degree, value):
    field, source = factory(characteristic, degree), source_factory(
        characteristic, degree
    )
    assert outcome(lambda: field.assert_valid_val(value, "input")) == outcome(
        lambda: source.assert_valid_val(value, "input")
    )
    assert outcome(lambda: field.to_bits(value)) == outcome(
        lambda: source.to_bits(value)
    )


@pytest.mark.parametrize("values", [[], [-1], [8], [0, 7], [0.5], [float("nan")]])
def test_array_range_only_validation(factory, values):
    field, source = factory(2, 3), source_factory(2, 3)
    array = np.asarray(values)
    assert outcome(lambda: field.assert_valid_val_array(array, "input")) == outcome(
        lambda: source.assert_valid_val_array(array, "input")
    )


@pytest.mark.parametrize("bits", [[], [0], [1], [1, 1, 1], [1, 0, 0, 0], [2], [-1]])
def test_unchecked_scalar_decode_and_field_range(factory, bits):
    field, source = factory(2, 3), source_factory(2, 3)
    assert outcome(lambda: field.from_bits(bits)) == outcome(
        lambda: source.from_bits(bits)
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_explicit_polynomial_identity_and_wrappers(factory, degree):
    import galois
    from qualtran import CGF, QGF

    polynomial = galois.GF(2, degree, compile="python-calculate").irreducible_poly
    field = factory(2, degree, polynomial)
    assert field.irreducible_poly is polynomial
    for wrapper in (QGF, CGF):
        source = wrapper(2, degree, polynomial)
        assert source.irreducible_poly is field.irreducible_poly
        assert source.order == field.order
        assert source.bitsize == field.bitsize
        assert source.gf_type is field.gf_type


def test_explicit_none_preserves_delayed_failure(factory):
    field = factory(4, 2, None)
    assert field.irreducible_poly is None
    assert field.order == 16
    assert field.bitsize == 4
    assert outcome(lambda: field.gf_type) == outcome(
        lambda: source_factory(4, 2, None).gf_type
    )
    assert outcome(lambda: factory(4, 2)) == outcome(lambda: source_factory(4, 2))
