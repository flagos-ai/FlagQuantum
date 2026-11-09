"""Original CAdd symbolic declaration dispatch, QEC profiles and width fallback."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.arithmetic.controlled_add_resource_model import (
    source_controlled_add_resource_formula,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def width_case(mode):
    m, n = sympy.symbols("m n", integer=True, positive=True)
    return {
        "concrete": (2, 3),
        "concrete_one": (1, 1),
        "symbolic": (m, n),
        "left_concrete": (1, n),
        "right_concrete": (m, 3),
        "literal_one": (sympy.Integer(1), sympy.Integer(1)),
        "literals": (sympy.Integer(2), sympy.Integer(3)),
        "reversed_literals": (sympy.Integer(3), sympy.Integer(2)),
        "left_literal": (sympy.Integer(2), 3),
        "right_literal": (2, sympy.Integer(3)),
    }[mode]


def assert_formula(formula, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for field in fields(formula):
        if field.name != "logical_qubits":
            assert (
                sympy.simplify(
                    getattr(formula, field.name) - getattr(expected, field.name)
                )
                == 0
            )
    assert (
        sympy.simplify(formula.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


@pytest.mark.parametrize(
    "mode,cv,legacy,dtype",
    tuple(
        product(
            (
                "concrete",
                "concrete_one",
                "symbolic",
                "left_concrete",
                "right_concrete",
                "literal_one",
                "literals",
                "reversed_literals",
                "left_literal",
                "right_literal",
            ),
            (0, 1),
            (False, True),
            ("uint", "montgomery"),
        )
    ),
)
def test_original_cadd_declarations_both_directions(mode, cv, legacy, dtype):
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    m, n = width_case(mode)
    formula = source_controlled_add_resource_formula(
        m, n, control_value=cv, legacy_shims=legacy
    )
    word = QUInt if dtype == "uint" else QMontgomeryUInt
    original = CAdd(word(m), word(n), cv=cv)
    assert_formula(formula, original, legacy)
    assert_formula(formula, original.adjoint(), legacy)


@pytest.mark.parametrize(
    "mode,legacy",
    tuple(
        product(
            (
                "concrete",
                "concrete_one",
                "symbolic",
                "left_concrete",
                "right_concrete",
                "literal_one",
                "literals",
                "left_literal",
                "right_literal",
            ),
            (False, True),
        )
    ),
)
def test_shared_add_formula_keeps_original_profile_and_dispatch(mode, legacy):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    m, n = width_case(mode)
    formula = source_add_resource_formula(m, n, legacy_shims=legacy)
    source = Add(QUInt(m), QUInt(n))
    assert_formula(formula, source, legacy)
    assert_formula(formula, source.adjoint(), legacy)


def test_shared_symbolic_children_are_replaceable_and_all_categories_propagate(
    monkeypatch,
):
    import flagquantum.compiler.registers.arithmetic.controlled_add_resource_model as module

    n = sympy.Symbol("n", positive=True, integer=True)
    add = SymbolicResourceFormula(2, 3, 5, 17, t=7, toffoli=11, cswap=13, rotation=19)
    compute = ComponentCost("predicate", 23, GateCost(and_bloq=1, clifford=2, t=2))
    cleanup = ComponentCost(
        "cleanup",
        29,
        GateCost(measurement=1, clifford=3, toffoli=3, cswap=5, rotation=7),
    )
    calls = []

    def addition(m, r, *, legacy_shims=False):
        assert m is n
        calls.append(("add", m, r, legacy_shims))
        return add

    def predicate(values, *, uncompute=False, legacy_shims=False):
        calls.append(("predicate", values, uncompute, legacy_shims))
        return cleanup if uncompute else compute

    monkeypatch.setattr(module, "source_add_resource_formula", addition)
    monkeypatch.setattr(module, "control_predicate_cost", predicate)
    actual = module.source_controlled_add_resource_formula(
        n, 4, control_value=0, legacy_shims=True
    )
    assert actual == SymbolicResourceFormula(
        2 + n,
        3 + 5 * n,
        5 + n,
        sympy.Max(29, n + 5),
        t=7 + 2 * n,
        toffoli=11 + 3 * n,
        cswap=13 + 5 * n,
        rotation=19 + 7 * n,
    )
    assert calls == [
        ("add", n, 4, True),
        ("predicate", (0, 1), False, True),
        ("predicate", (0, 1), True, True),
    ]


def test_concrete_dispatch_uses_composed_peak_instead_of_root_ports(monkeypatch):
    import flagquantum.compiler.registers.arithmetic.controlled_add_resource_model as module

    child = ComponentCost("child", 29, GateCost(t=3))
    root = ComponentCost("root", 11, calls=(CostCall(child, 2, 3),))
    calls = []

    def concrete(m, n, **kwargs):
        calls.append((m, n, kwargs))
        return root

    monkeypatch.setattr(module, "source_controlled_add_cost", concrete)
    formula = module.source_controlled_add_resource_formula(
        2, 3, control_value=0, legacy_shims=True
    )
    assert formula.t == 6 and formula.logical_qubits == 32
    assert calls == [(2, 3, dict(control_value=0, legacy_shims=True))]


def test_symbolic_substitution_does_not_invent_concrete_loaded_and_carry_lifetimes():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QubitCount, get_cost_value

    n = sympy.Symbol("n", integer=True, positive=True)
    formula = source_controlled_add_resource_formula(n, n)
    literal = source_controlled_add_resource_formula(sympy.Integer(3), sympy.Integer(3))
    concrete = source_controlled_add_resource_formula(3, 3)
    assert formula.logical_qubits.subs(n, 3) == literal.logical_qubits == 7
    assert concrete.logical_qubits == 12
    assert get_cost_value(CAdd(QUInt(n)), QubitCount()).subs(n, 3) == 7
    assert get_cost_value(CAdd(QUInt(3)), QubitCount()) == 12
    for field in fields(formula):
        if field.name != "logical_qubits":
            assert sympy.sympify(getattr(formula, field.name)).subs(n, 3) == getattr(
                concrete, field.name
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_width": True},
        {"left_width": 0},
        {"right_width": 1.5},
        {"control_value": True},
        {"control_value": 2},
        {"legacy_shims": 0},
    ],
)
def test_formula_parameters_are_explicit(kwargs):
    with pytest.raises(ValueError):
        source_controlled_add_resource_formula(
            **(dict(left_width=2, right_width=3) | kwargs)
        )


@pytest.mark.parametrize("legacy", [None, 0, 1, "legacy"])
def test_shared_add_profile_requires_boolean(legacy):
    with pytest.raises(ValueError):
        source_add_resource_formula(2, 3, legacy_shims=legacy)


@pytest.mark.parametrize("value", [None, "3"])
def test_unknown_width_objects_do_not_silently_enter_symbolic_dispatch(value):
    with pytest.raises(ValueError, match="Python integers or symbolic expressions"):
        source_controlled_add_resource_formula(value, 3)


def test_numpy_width_requires_explicit_native_integer_conversion():
    import numpy as np

    with pytest.raises(ValueError, match="Python integers or symbolic expressions"):
        source_controlled_add_resource_formula(np.int64(3), 3)


def test_symbolic_formula_consumes_replaceable_native_call_recipe(monkeypatch):
    import flagquantum.compiler.registers.arithmetic.controlled_add_resource_model as module
    from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    n = sympy.Symbol("n", integer=True, positive=True)
    seen = []

    def recipe(root):
        seen.append(root)
        return {FixedTCall("fredkin"): 2 * n}

    monkeypatch.setattr(module, "arithmetic_callees", recipe)
    actual = module.source_controlled_add_resource_formula(n, n, control_value=0)
    assert seen == [ArithmeticCall("controlled_add", (n, n), control_value=0)]
    assert actual.cswap == 2 * n
    assert actual.and_bloq == actual.clifford == actual.measurement == 0
    assert actual.logical_qubits == sympy.Max(3, 2 * n + 1)
