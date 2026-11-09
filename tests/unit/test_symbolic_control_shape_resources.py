"""Source unknown-polarity controls and whole-wrapper symbolic width fallback."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler.registers.arithmetic.grouped_constant_add_resource_model import (
    source_pattern_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.bits.xor_resource_model import (
    source_pattern_controlled_xor_resource_formula,
)
from flagquantum.compiler.registers.control.control_predicate_resource_model import (
    control_predicate_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n, m, k = sympy.symbols("n m k", positive=True, integer=True)
z = sympy.Symbol("z", integer=True)
CASES = (
    ("QBit", 1, (2,), 2),
    ("QBit", 1, (3,), 3),
    ("QBit", 1, (sympy.Integer(2),), sympy.Integer(2)),
    ("QBit", 1, (sympy.Integer(3),), sympy.Integer(3)),
    ("QBit", 1, (sympy.Integer(0),), sympy.Integer(0)),
    ("QBit", 1, (m,), m),
    ("QBit", 1, (z,), z),
    ("QBit", 1, (m + 2,), m + 2),
    ("QBit", 1, (2, m), 2 * m),
    ("QUInt", m, (), m),
    ("QInt", m, (2,), 2 * m),
)
DIMENSIONS = (
    (3, 5),
    (sympy.Integer(3), sympy.Integer(5)),
    (n, sympy.Integer(5)),
    (3, k),
    (n, k),
)


def spec(case):
    import qualtran
    from qualtran.symbolics import Shaped

    name, width, shape, _ = case
    dtype = qualtran.QBit() if name == "QBit" else getattr(qualtran, name)(width)
    return qualtran.CtrlSpec(qdtypes=dtype, cvs=Shaped(shape))


def compare(actual, original, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    expected = get_cost_value(original, QECGatesCost(legacy_shims=legacy))
    for field in fields(actual):
        value = (
            get_cost_value(original, QubitCount())
            if field.name == "logical_qubits"
            else getattr(expected, field.name)
        )
        assert sympy.simplify(getattr(actual, field.name) - value) == 0


@pytest.mark.parametrize(
    "case,adjoint,legacy", tuple(product(CASES, (False, True), (False, True)))
)
def test_unknown_shaped_predicate_counts_and_original_peak(case, adjoint, legacy):
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    source = CtrlSpecAnd(spec(case))
    if adjoint:
        source = source.adjoint()
    actual = control_predicate_resource_formula(
        case[3], uncompute=adjoint, legacy_shims=legacy
    )
    compare(actual, source, legacy)


@pytest.mark.parametrize(
    "case,dimensions,kind,adjoint,legacy",
    tuple(product(CASES, DIMENSIONS, ("xor", "add"), (False, True), (False, True))),
)
def test_two_consumers_and_original_symbolic_shape_fallback(
    case, dimensions, kind, adjoint, legacy
):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.bloqs.arithmetic.bitwise import Xor

    width, constant = dimensions
    body = Xor(QUInt(width)) if kind == "xor" else AddK(QUInt(width), constant)
    source = body.controlled(spec(case))
    if adjoint:
        source = source.adjoint()
    actual = (
        source_pattern_controlled_xor_resource_formula(
            width, case[3], legacy_shims=legacy, control_shapes=(case[2],)
        )
        if kind == "xor"
        else source_pattern_controlled_constant_add_resource_formula(
            width,
            constant,
            case[3],
            adjoint=adjoint,
            legacy_shims=legacy,
            control_shapes=(case[2],),
        )
    )
    compare(actual, source, legacy)


def test_literal_control_count_uses_source_wrapper_dispatch_not_integer_coercion():
    concrete = source_pattern_controlled_xor_resource_formula(3, 2)
    literal = source_pattern_controlled_xor_resource_formula(3, sympy.Integer(2))
    assert concrete.toffoli == literal.toffoli == 3
    assert (concrete.logical_qubits, literal.logical_qubits) == (9, 8)
    add = source_pattern_controlled_constant_add_resource_formula(3, 5, 2)
    add_literal = source_pattern_controlled_constant_add_resource_formula(
        3, 5, sympy.Integer(2)
    )
    assert (add.logical_qubits, add_literal.logical_qubits) == (11, 9)


@pytest.mark.parametrize(
    "count,kind", tuple(product((0, 1, sympy.Integer(1)), ("xor", "add")))
)
def test_unknown_zero_or_single_control_failures_are_retained(count, kind):
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.symbolics import Shaped

    source = (Xor(QUInt(3)) if kind == "xor" else AddK(QUInt(3), 5)).controlled(
        CtrlSpec(cvs=Shaped((count,)))
    )
    with pytest.raises(ValueError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(ValueError):
        if kind == "xor":
            source_pattern_controlled_xor_resource_formula(3, count)
        else:
            source_pattern_controlled_constant_add_resource_formula(3, 5, count)


@pytest.mark.parametrize("pattern", ((), (2,), (True,), 0, 1, False, 2.5))
def test_pattern_contract_rejects_nonbinary_or_invalid_counts(pattern):
    with pytest.raises(ValueError):
        source_pattern_controlled_xor_resource_formula(3, pattern)


def test_same_symbolic_count_with_array_shape_or_one_word_keeps_different_lifetimes():
    array = source_pattern_controlled_xor_resource_formula(3, m, control_shapes=((m,),))
    word = source_pattern_controlled_xor_resource_formula(3, m, control_shapes=((),))
    assert array.logical_qubits.subs(m, 3) == 9
    assert word.logical_qubits.subs(m, 3) == 11


@pytest.mark.parametrize("shapes", ((), ((True,),), ((-1,),), ([2],)))
def test_control_shape_metadata_cannot_silently_change_dispatch(shapes):
    with pytest.raises(ValueError):
        source_pattern_controlled_xor_resource_formula(3, m, control_shapes=shapes)


@pytest.mark.parametrize("shape", ((m,), ()))
def test_shared_predicate_replacement_reaches_both_shape_dispatches(monkeypatch, shape):
    import flagquantum.compiler.registers.control.controlled_component_resource_model as shared
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    seen = []

    def predicate(controls, *, uncompute=False, legacy_shims=False):
        assert controls is m and legacy_shims
        seen.append(uncompute)
        return (
            SymbolicResourceFormula(
                31, 37, 43, 43, t=19, toffoli=23, cswap=29, rotation=41
            )
            if uncompute
            else SymbolicResourceFormula(
                7, 11, 17, 37, t=2, toffoli=3, cswap=5, rotation=13
            )
        )

    monkeypatch.setattr(shared, "control_predicate_resource_formula", predicate)
    actual = source_pattern_controlled_xor_resource_formula(
        3, m, control_shapes=(shape,), legacy_shims=True
    )
    assert seen == [False, True]
    peak = sympy.Max(43, m + 6) if shape else sympy.Max(49, 2 * m + 5)
    assert actual == SymbolicResourceFormula(
        38, 48, 60, peak, t=21, toffoli=29, cswap=34, rotation=54
    )
