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
        assert outcome(
            lambda value=value: field.from_bits(source.to_bits(value))
        ) == outcome(lambda value=value: source.from_bits(source.to_bits(value)))
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


@pytest.mark.parametrize("polynomial", [OMITTED, None, [3, 1, 0]])
@pytest.mark.parametrize("symbolic", [False, True])
def test_symbolic_protocol_and_field_factory_call_order(
    factory, monkeypatch, polynomial, symbolic
):
    import galois

    events = []

    class Characteristic(int):
        def is_symbolic(self):
            events.append("characteristic symbolic")
            return symbolic

    class Degree(int):
        def is_symbolic(self):
            events.append("degree symbolic")
            return False

    original = galois.GF

    def traced(*args, **kwargs):
        events.append((args, kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(galois, "GF", traced)
    characteristic, degree = Characteristic(2), Degree(3)
    source = source_factory(characteristic, degree, polynomial)
    for name in ("order", "bitsize", "gf_type"):
        outcome(lambda name=name: getattr(source, name))
        outcome(lambda name=name: getattr(source, name))
    expected = events.copy()
    events.clear()
    field = factory(characteristic, degree, polynomial)
    for name in ("order", "bitsize", "gf_type"):
        outcome(lambda name=name: getattr(field, name))
        outcome(lambda name=name: getattr(field, name))
    assert events == expected


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (2, 3), (2, 2), (2, 4), (1, 2, 3)])
def test_array_decode_shapes_and_diagnostics(factory, shape):
    field, source = factory(2, 3), source_factory(2, 3)
    bits = np.zeros(shape, dtype=np.uint8)
    assert outcome(lambda: field.from_bits_array(bits)) == outcome(
        lambda: source.from_bits_array(bits)
    )


def test_prime_field_ignores_explicit_extension_polynomial(factory):
    import galois

    polynomial = galois.Poly([1, 2, 1], field=galois.GF(3))
    field = factory(5, 1, polynomial)
    source = source_factory(5, 1, polynomial)
    assert field.irreducible_poly is polynomial
    assert field.gf_type is source.gf_type
    assert field.gf_type.order == 5


@pytest.mark.parametrize("degree,modulus", [(2, 7), (3, 11)])
def test_numerical_binding_replacement_in_append_multiplication(
    factory, degree, modulus
):
    import galois
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.finite_field.field_multiplication import (
        bind_field_outputs,
        gf2_multiplication,
    )

    polynomial = galois.Poly.Int(modulus)
    source = source_factory(2, degree, polynomial)
    field = factory(2, degree, polynomial)
    registers = tuple(tuple(range(i * degree, (i + 1) * degree)) for i in range(3))

    def instructions(encoding):
        plan = gf2_multiplication(
            *registers,
            (),
            modulus=int(encoding.irreducible_poly),
            algorithm="quadratic",
        )
        return bind_field_outputs(plan, registers)

    gates = instructions(field)
    assert gates == instructions(source)
    from collections import Counter

    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_field_multiplication import (
        assert_gate_streams,
        reference_gates,
    )

    plan = gf2_multiplication(*registers, (), modulus=modulus, algorithm="quadratic")
    upstream = GF2Multiplication(QGF(2, degree, polynomial))
    original_gates, original_outputs = reference_gates(
        upstream, {"x": registers[0], "y": registers[1]}, registers[2]
    )
    assert_gate_streams(
        [(gate.name, gate.wires) for gate in plan.instructions], original_gates
    )
    assert plan.outputs == (
        original_outputs["x"],
        original_outputs["y"],
        original_outputs["result"],
    )
    emitted = Counter(gate.name for gate in plan.instructions)
    model = get_cost_value(upstream, QECGatesCost())
    assert (emitted["ccx"], emitted["cx"]) == (model.toffoli, model.clifford)
    count = 1 << degree
    generator = torch.Generator().manual_seed(73)
    amplitudes = torch.randn(count * count, dtype=torch.complex128, generator=generator)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(count**3, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for left in range(count):
        for right in range(count):
            value = amplitudes[left * count + right]
            index = (left * count + right) * count
            state[index] = value
            bits = field.to_bits(field.gf_type(left) * field.gf_type(right))
            product = int("".join(map(str, bits)), 2)
            expected[index + product] = value
    result = (
        Circuit(3 * degree, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(3 * degree, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_shared_unsigned_decoder_can_be_replaced(monkeypatch):
    from qualtran._infra.data_types import _UInt

    import flagquantum.simulation.finite_field_encoding as native

    field = native.FiniteFieldEncoding(2, 3)
    bits = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    expected = field.from_bits_array(bits)
    calls = []

    def source_decode(bits, width):
        calls.append(width)
        return _UInt(width).from_bits_array(bits)

    monkeypatch.setattr(native, "integer_array_from_bits", source_decode)
    np.testing.assert_array_equal(field.from_bits_array(bits), expected)
    assert calls == [3]


@pytest.mark.parametrize("polynomial", [OMITTED, None])
def test_missing_optional_dependency_is_observable(factory, monkeypatch, polynomial):
    import sys

    monkeypatch.setitem(sys.modules, "galois", None)
    with pytest.raises(ModuleNotFoundError, match="galois"):
        factory(N, 2, polynomial)


def test_different_polynomial_field_scalars_remain_nominal(factory):
    import galois

    field, source = factory(2, 3), source_factory(2, 3)
    other = galois.GF(
        2, 3, irreducible_poly=galois.Poly.Int(13), compile="python-calculate"
    )
    value = other(3)
    assert outcome(lambda: field.to_bits(value)) == outcome(
        lambda: source.to_bits(value)
    )
    assert outcome(lambda: field.assert_valid_val(value))[0] is ValueError
