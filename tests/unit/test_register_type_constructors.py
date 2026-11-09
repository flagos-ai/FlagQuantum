"""Replace source-backed metadata constructors without changing consumers."""

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)


def source_factory(kind, width=OMITTED, bound=OMITTED):
    from qualtran import BQUInt, QAny, QBit, QUInt

    if kind == "bit":
        dtype = QBit()
    elif kind == "bounded":
        dtype = BQUInt(width) if bound is OMITTED else BQUInt(width, bound)
    else:
        dtype = {"opaque": QAny, "uint": QUInt}[kind](width)
    return dtype


def source_contract(kind, width=OMITTED, bound=OMITTED):
    dtype = source_factory(kind, width, bound)
    return RegisterType(
        {"bit": "bit", "bounded": "uint", "opaque": "opaque", "uint": "uint"}[kind],
        1 if kind == "bit" else dtype.bitsize,
        parameters=(("bound", dtype.iteration_length),) if kind == "bounded" else (),
    )


@pytest.fixture(params=["source_contract"])
def construct_type(request):
    return source_contract


def outcome(factory, kind, width, bound=OMITTED):
    try:
        value = factory(kind, width, bound)
    except (TypeError, ValueError) as error:
        message = (
            str(error)
            .replace("QAny", "opaque register")
            .replace("BQUInt", "Bounded unsigned register")
        )
        return type(error), message
    return value.encoding, value.quantum_bits, value.classical_bits, value.parameters


@pytest.mark.parametrize("kind", ["opaque", "uint", "bounded"])
@pytest.mark.parametrize(
    "width",
    [-2, -1, 0, 1, 2, True, False, np.int64(2), sympy.Integer(2), N, 1.5, None, "3"],
)
def test_width_construction_matches_original_outcomes(construct_type, kind, width):
    assert outcome(construct_type, kind, width) == outcome(source_contract, kind, width)


@pytest.mark.parametrize("width", [-1, 0, 1, 3, np.int64(3), sympy.Integer(3), N])
@pytest.mark.parametrize(
    "bound", [-1, 0, 1, 3, 8, 9, True, None, 1.5, sympy.Integer(8), N]
)
def test_explicit_bounded_construction_matches_original(construct_type, width, bound):
    assert outcome(construct_type, "bounded", width, bound) == outcome(
        source_contract, "bounded", width, bound
    )


@pytest.mark.parametrize(
    "kind,width", [("bit", 1), ("opaque", 0), ("opaque", -1), ("uint", 0), ("uint", 3)]
)
def test_constructor_identity_and_signature_consumer(construct_type, kind, width):
    dtype = construct_type(kind, width)
    assert dtype == RegisterType(kind, width)
    assert type(dtype.quantum_bits) is int
    register = Register("word", dtype, shape=(2, 3))
    signature = Signature((register,))
    assert signature.get_left("word").dtype is dtype
    assert signature.get_right("word").dtype is dtype
    assert signature.n_qubits() == 6 * width
    assert list(register.all_idxs()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_default_and_explicit_bound_are_distinct(construct_type):
    dtype = construct_type("bounded", 3)
    assert dtype == RegisterType("uint", 3, parameters=(("bound", 8),))
    with pytest.raises(TypeError):
        construct_type("bounded", 3, None)
    assert construct_type("bounded", N).parameters == (("bound", 2**N),)
    # The source accepts an explicit None bound when width is symbolic.
    assert construct_type("bounded", N, None).parameters == (("bound", None),)


def test_default_bound_evaluates_before_symbolic_guard(construct_type):
    events = []

    class Width:
        def __rpow__(self, base):
            events.append(("power", base))
            return 7

        def is_symbolic(self):
            events.append("symbolic")
            return True

    width = Width()
    dtype = construct_type("bounded", width)
    assert dtype.quantum_bits is width
    assert dtype.parameters == (("bound", 7),)
    assert events == [("power", 2), "symbolic"]


def test_bounded_guard_short_circuits_later_checker(construct_type):
    class UntouchedBound:
        def is_symbolic(self):
            raise AssertionError(
                "Symbolic width should short-circuit the bound checker"
            )

    bound = UntouchedBound()
    dtype = construct_type("bounded", N, bound)
    assert dtype.parameters[0][1] is bound
