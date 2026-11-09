"""Source-backed and native fixed-point metadata use the same consumer cases."""

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)
F = sympy.Symbol("f", integer=True, positive=True)
SHAPES = [
    (3, 2),
    (1, 0),
    (1, 1),
    (0, 0),
    (0, -1),
    (-1, -2),
    (3, 3),
    (3, 4),
    (3, -2),
    (True, False),
    (False, False),
    (np.int64(1), np.int64(0)),
    (np.int64(3), np.int64(2)),
    (sympy.Integer(1), sympy.Integer(0)),
    (sympy.Integer(3), sympy.Integer(2)),
    (N, 2),
    (3, F),
    (N, F),
    (3.0, 2),
    (3, 2.0),
    (None, 2),
    (3, None),
    ("3", 2),
    (3, "2"),
    (65, 2),
    (1, F),
]


def original_type(classical, width, fractional_width, signed=OMITTED):
    from qualtran import CFxp, QFxp

    factory = CFxp if classical else QFxp
    if signed is OMITTED:
        return factory(width, fractional_width)
    return factory(width, fractional_width, signed)


def source_contract(classical, width, fractional_width, signed=OMITTED):
    source = original_type(classical, width, fractional_width, signed)
    parameters = (("storage", "classical"),) if classical else ()
    parameters += (("fractional_width", source.num_frac), ("signed", source.signed))
    return RegisterType(
        "fixed_point",
        0 if classical else source.bitsize,
        source.bitsize if classical else 0,
        parameters=parameters,
    )


def native_factory(classical, width, fractional_width, signed=OMITTED):
    from flagquantum.core.ir.program.register_types import (
        classical_fixed_point_register_type,
        fixed_point_register_type,
    )

    factory = (
        classical_fixed_point_register_type if classical else fixed_point_register_type
    )
    if signed is OMITTED:
        return factory(width, fractional_width)
    return factory(width, fractional_width, signed)


@pytest.fixture(params=["source_contract", "native"])
def construct_type(request):
    return source_contract if request.param == "source_contract" else native_factory


def outcome(factory, classical, width, fractional_width, signed):
    try:
        value = factory(classical, width, fractional_width, signed)
    except (TypeError, ValueError) as error:
        return type(error), str(error).replace("QFxp", "fixed-point register")
    return (
        value.encoding,
        value.quantum_bits,
        type(value.quantum_bits),
        value.classical_bits,
        type(value.classical_bits),
        tuple(
            (name, parameter, type(parameter)) for name, parameter in value.parameters
        ),
    )


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width,fractional_width", SHAPES)
@pytest.mark.parametrize(
    "signed", [OMITTED, False, True, None, np.bool_(True), 1, sympy.Integer(1), F]
)
def test_constructor_values_types_and_eager_failures(
    construct_type, classical, width, fractional_width, signed
):
    assert outcome(
        construct_type, classical, width, fractional_width, signed
    ) == outcome(source_contract, classical, width, fractional_width, signed)


@pytest.mark.parametrize("classical", [False, True])
def test_default_false_and_zero_width_storage_identity(construct_type, classical):
    assert construct_type(classical, 3, 2) == construct_type(classical, 3, 2, False)
    assert construct_type(classical, 0, 0) != construct_type(not classical, 0, 0)
    assert construct_type(classical, 3, 2) != construct_type(classical, 3, 1)
    assert construct_type(classical, 3, 2) != construct_type(classical, 3, 2, True)


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width,fractional_width", [(0, 0), (3, 2), (N, 2), (3, F)])
def test_shaped_signature_widths_match_original(
    construct_type, classical, width, fractional_width
):
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = construct_type(classical, width, fractional_width)
    signature = Signature((Register("word", dtype, shape=(2, 3)),))
    source = SourceSignature(
        (
            SourceRegister(
                "word", original_type(classical, width, fractional_width), shape=(2, 3)
            ),
        )
    )
    assert signature.n_qubits() == source.n_qubits()
    assert signature.n_cbits() == source.n_cbits()
    assert signature.n_bits() == source.n_bits()
    assert signature.get_left("word").dtype is dtype


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
def test_original_protocol_call_order_and_symbolic_short_circuit(
    construct_type, classical, symbolic
):
    events = []

    class Width(int):
        def is_symbolic(self):
            events.append("width symbolic")
            return symbolic

        def __eq__(self, other):
            events.append(("width equals", other))
            return super().__eq__(other)

        def __lt__(self, other):
            events.append(("width less", other))
            return super().__lt__(other)

    class Fraction(int):
        def is_symbolic(self):
            events.append("fraction symbolic")
            return False

    width, fractional_width = Width(3), Fraction(2)
    source_contract(classical, width, fractional_width, True)
    expected = events.copy()
    events.clear()
    actual = construct_type(classical, width, fractional_width, True)
    assert events == expected
    assert (actual.classical_bits if classical else actual.quantum_bits) is width
    assert actual.parameters[-2][1] is fractional_width
