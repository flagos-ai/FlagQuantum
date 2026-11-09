"""Montgomery constructor metadata retains original unvalidated field identity."""

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)
P = sympy.Symbol("p", integer=True, positive=True)


def original_type(classical, width, modulus=OMITTED):
    from qualtran import CMontgomeryUInt, QMontgomeryUInt

    factory = CMontgomeryUInt if classical else QMontgomeryUInt
    return factory(width) if modulus is OMITTED else factory(width, modulus)


def source_contract(classical, width, modulus=OMITTED):
    source = original_type(classical, width, modulus)
    parameters = (("storage", "classical"),) if classical else ()
    if source.modulus is not None:
        parameters += (("modulus", source.modulus),)
    return RegisterType(
        "montgomery",
        0 if classical else source.bitsize,
        source.bitsize if classical else 0,
        parameters=parameters,
    )


def native_factory(classical, width, modulus=OMITTED):
    from flagquantum.core.ir.program.register_types import (
        classical_montgomery_register_type,
        montgomery_register_type,
    )

    factory = (
        classical_montgomery_register_type if classical else montgomery_register_type
    )
    return factory(width) if modulus is OMITTED else factory(width, modulus)


@pytest.fixture(params=["source_contract", "native"])
def construct_type(request):
    return source_contract if request.param == "source_contract" else native_factory


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize(
    "width",
    [-2, -1, 0, 1, 3, True, False, np.int64(3), sympy.Integer(3), N, 1.5, None, "3"],
)
@pytest.mark.parametrize(
    "modulus",
    [OMITTED, None, -5, 0, 1, 5, 6, True, np.int64(5), sympy.Integer(5), P, 5.0],
)
def test_declared_fields_match_source_without_numeric_validation(
    construct_type, classical, width, modulus
):
    expected = source_contract(classical, width, modulus)
    actual = construct_type(classical, width, modulus)
    assert actual == expected
    assert type(actual.quantum_bits) is type(expected.quantum_bits)
    assert type(actual.classical_bits) is type(expected.classical_bits)
    assert actual.parameters == expected.parameters
    assert tuple(type(value) for _, value in actual.parameters) == tuple(
        type(value) for _, value in expected.parameters
    )


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width", [0, 3, N])
def test_default_none_identity_and_modulus_identity(construct_type, classical, width):
    assert construct_type(classical, width) == construct_type(classical, width, None)
    assert construct_type(classical, width, 5) != construct_type(classical, width, 7)
    assert construct_type(classical, width, 5) != construct_type(
        not classical, width, 5
    )


@pytest.mark.parametrize("classical", [False, True])
def test_constructor_does_not_evaluate_fields(construct_type, classical):
    class Unevaluated:
        def __bool__(self):
            raise AssertionError(
                "Constructor must retain the field without a truth test"
            )

        def is_symbolic(self):
            raise AssertionError("Constructor must not classify symbolic fields")

        def __int__(self):
            raise AssertionError("Constructor must not coerce fields")

    width, modulus = Unevaluated(), Unevaluated()
    actual = construct_type(classical, width, modulus)
    assert (actual.classical_bits if classical else actual.quantum_bits) is width
    assert actual.parameters[-1][1] is modulus


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width", [0, 3, N])
def test_shaped_signature_consumes_montgomery_width(construct_type, classical, width):
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = construct_type(classical, width, P)
    signature = Signature((Register("word", dtype, shape=(2, 3)),))
    source = SourceSignature(
        (SourceRegister("word", original_type(classical, width, P), shape=(2, 3)),)
    )
    assert signature.n_qubits() == source.n_qubits()
    assert signature.n_cbits() == source.n_cbits()
    assert signature.n_bits() == source.n_bits()
    assert signature.get_left("word").dtype is dtype


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width", [0, 3, N, np.int64(3), sympy.Integer(3)])
@pytest.mark.parametrize("modulus", [None, 5, P])
def test_montgomery_symbolic_classification_ignores_modulus(classical, width, modulus):
    source = original_type(classical, width, modulus)
    actual = native_factory(classical, width, modulus)
    assert actual.is_symbolic() == source.is_symbolic()
    assert Register("word", actual).is_symbolic() == source.is_symbolic()
    generic = RegisterType("uint", 3, parameters=(("bound", P),))
    assert generic.is_symbolic()


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width,modulus", [(3, 3), (4, 5), (5, 9), (6, 13)])
def test_existing_numeric_consumers_accept_replaced_descriptor(
    construct_type, classical, width, modulus
):
    from math import gcd

    from flagquantum.simulation.montgomery_values import (
        integer_to_montgomery_value,
        montgomery_inverse_value,
        montgomery_product_values,
        montgomery_to_integer_value,
        montgomery_value_domain,
    )

    descriptor = construct_type(classical, width, modulus)
    source = original_type(classical, width, modulus)._bit_encoding
    declared_width = descriptor.classical_bits if classical else descriptor.quantum_bits
    declared_modulus = dict(descriptor.parameters)["modulus"]
    assert (
        montgomery_value_domain(declared_width, declared_modulus) == source.get_domain()
    )
    for left in range(modulus):
        stored = integer_to_montgomery_value(left, declared_width, declared_modulus)
        assert stored == source.uint_to_montgomery(left)
        assert (
            montgomery_to_integer_value(stored, declared_width, declared_modulus)
            == left
        )
        for right in range(modulus):
            other = integer_to_montgomery_value(right, declared_width, declared_modulus)
            actual = montgomery_product_values(
                stored, other, declared_width, declared_modulus
            )
            assert actual == source.montgomery_product(stored, other)
            assert (
                montgomery_to_integer_value(actual, declared_width, declared_modulus)
                == (left * right) % modulus
            )
        if gcd(left, modulus) == 1:
            assert montgomery_inverse_value(
                stored, declared_width, declared_modulus
            ) == source.montgomery_inverse(stored)


@pytest.mark.parametrize("classical", [False, True])
def test_symbolic_modulus_construction_precedes_numeric_failure(
    construct_type, classical
):
    from flagquantum.simulation.montgomery_values import integer_to_montgomery_value

    descriptor = construct_type(classical, 3, P)
    assert descriptor.parameters[-1] == ("modulus", P)
    with pytest.raises(AssertionError):
        integer_to_montgomery_value(1, descriptor.num_bits, P)
    with pytest.raises(AssertionError):
        original_type(classical, 3, P)._bit_encoding.uint_to_montgomery(1)


@pytest.mark.parametrize("classical", [False, True])
def test_native_construction_without_optional_sympy(classical, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sympy" or name.startswith("sympy."):
            raise ImportError("Optional Sympy deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    descriptor = native_factory(classical, 3, 5)
    assert descriptor.num_bits == 3
    assert descriptor.parameters[-1] == ("modulus", 5)
