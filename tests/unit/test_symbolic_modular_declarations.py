"""Original symbolic modular signatures and formulas retain literal dispatch."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler.registers.modular_declarations import (
    modular_arithmetic_calls,
    modular_arithmetic_registers,
    source_modular_arithmetic_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def dimension(mode):
    if mode == "positive":
        return sympy.Symbol("n", integer=True, positive=True)
    if mode == "nonnegative":
        return sympy.Symbol("n", integer=True, nonnegative=True)
    if mode == "integer":
        return sympy.Symbol("n", integer=True)
    if mode == "expression":
        return sympy.Symbol("n", integer=True, positive=True) + 1
    return {
        "zero": sympy.Integer(0),
        "one": sympy.Integer(1),
        "three": sympy.Integer(3),
        "python_three": 3,
    }[mode]


def original(kind, width, signed=False):
    from qualtran.bloqs.arithmetic._shims import CHalf, CSub, Lt, MultiCToffoli

    factories = {
        "multi_controlled_x": MultiCToffoli,
        "controlled_subtraction": CSub,
        "less_than": Lt,
        "controlled_half": CHalf,
    }
    return (
        factories[kind](width, signed=signed)
        if kind == "less_than"
        else factories[kind](width)
    )


@pytest.mark.parametrize(
    "kind,mode,legacy",
    tuple(
        product(
            (
                "multi_controlled_x",
                "controlled_subtraction",
                "less_than",
                "controlled_half",
            ),
            (
                "positive",
                "nonnegative",
                "integer",
                "expression",
                "zero",
                "one",
                "three",
                "python_three",
            ),
            (False, True),
        )
    ),
)
def test_original_all_formula_fields_and_symbolic_registers(kind, mode, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    width = dimension(mode)
    source = original(kind, width)
    assert modular_arithmetic_registers(kind, width) == tuple(
        (r.name, r.total_bits()) for r in source.signature
    )
    if kind == "controlled_subtraction" and mode == "zero":
        with pytest.raises(TypeError):
            get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        with pytest.raises(RuntimeError):
            get_cost_value(source, QubitCount())
        with pytest.raises(ValueError, match="zero-width CSub"):
            source_modular_arithmetic_resource_formula(kind, width, legacy_shims=legacy)
        return
    formula = source_modular_arithmetic_resource_formula(
        kind, width, legacy_shims=legacy
    )
    for operation in (source, source.adjoint()):
        expected = get_cost_value(operation, QECGatesCost(legacy_shims=legacy))
        for field in fields(formula):
            if field.name != "logical_qubits":
                assert (
                    sympy.simplify(
                        getattr(formula, field.name) - getattr(expected, field.name)
                    )
                    == 0
                )
        assert (
            sympy.simplify(
                formula.logical_qubits - get_cost_value(operation, QubitCount())
            )
            == 0
        )


@pytest.mark.parametrize(
    "mode,legacy",
    tuple(product(("positive", "integer", "one", "three"), (False, True))),
)
def test_original_signed_flag_retains_the_same_symbolic_declaration(mode, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    width = dimension(mode)
    signed = original("less_than", width, True)
    formula = source_modular_arithmetic_resource_formula(
        "less_than", width, signed=True, legacy_shims=legacy
    )
    assert formula == source_modular_arithmetic_resource_formula(
        "less_than", width, legacy_shims=legacy
    )
    assert (
        sympy.simplify(
            formula.toffoli
            - get_cost_value(signed, QECGatesCost(legacy_shims=legacy)).toffoli
        )
        == 0
    )
    assert (
        sympy.simplify(formula.logical_qubits - get_cost_value(signed, QubitCount()))
        == 0
    )


def test_csub_consumes_shared_symbolic_add_and_existing_on_each_query(monkeypatch):
    import flagquantum.compiler.registers.modular_declarations as module
    from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
        OnEachCall,
    )
    from flagquantum.compiler.registers.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    n = sympy.Symbol("n", integer=True, positive=True)
    addition = SymbolicResourceFormula(
        2, 3, 5, 29, t=7, toffoli=11, cswap=13, rotation=17
    )
    calls = []

    def add(left, right, *, legacy_shims=False):
        assert left is right is n
        calls.append(("add", legacy_shims))
        return addition

    def query(node, *, legacy_shims=False):
        assert node == OnEachCall(CliffordCall("x", 1), n)
        calls.append(("query", legacy_shims))
        return QECGateCost(clifford=2, t=3)

    monkeypatch.setattr(module, "source_controlled_add_resource_formula", add)
    monkeypatch.setattr(module, "query_qec_gate_cost", query)
    formula = module.source_modular_arithmetic_resource_formula(
        "controlled_subtraction", n, legacy_shims=True
    )
    assert formula == SymbolicResourceFormula(
        2, 9, 5, sympy.Max(29, 2 * n + 1, n), t=16, toffoli=11, cswap=13, rotation=17
    )
    assert calls == [("add", True), ("query", True)]


def test_source_symbolic_peak_substitution_differs_from_concrete_reconstruction():
    from qualtran.resource_counting import QubitCount, get_cost_value

    n = sympy.Symbol("n", integer=True, positive=True)
    formula = source_modular_arithmetic_resource_formula("controlled_subtraction", n)
    literal = source_modular_arithmetic_resource_formula(
        "controlled_subtraction", sympy.Integer(3)
    )
    concrete = source_modular_arithmetic_resource_formula("controlled_subtraction", 3)
    assert formula.logical_qubits.subs(n, 3) == literal.logical_qubits == 7
    assert concrete.logical_qubits == 12
    assert (
        get_cost_value(original("controlled_subtraction", n), QubitCount()).subs(n, 3)
        == 7
    )
    assert get_cost_value(original("controlled_subtraction", 3), QubitCount()) == 12
    for field in fields(formula):
        if field.name != "logical_qubits":
            assert sympy.sympify(getattr(formula, field.name)).subs(n, 3) == getattr(
                concrete, field.name
            )


def test_raw_concrete_callee_contract_does_not_cast_symbolic_dimensions():
    for width in (dimension("positive"), sympy.Integer(3)):
        with pytest.raises(ValueError, match="Python integer widths"):
            modular_arithmetic_calls("controlled_subtraction", width)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signed": 0},
        {"legacy_shims": 0},
        {"kind": "controlled_half", "signed": True},
        {"width": 1.5},
        {"width": None},
        {"kind": "unknown"},
    ],
)
def test_symbolic_options_keep_explicit_validation(kwargs):
    options = dict(kind="less_than", width=dimension("positive")) | kwargs
    with pytest.raises(ValueError):
        source_modular_arithmetic_resource_formula(**options)


@pytest.mark.parametrize(
    "kind,width,legacy",
    tuple(
        product(
            (
                "multi_controlled_x",
                "controlled_subtraction",
                "less_than",
                "controlled_half",
            ),
            (3, sympy.Symbol("n", integer=True, positive=True)),
            (False, True),
        )
    ),
)
def test_all_modular_formula_consumers_use_replaceable_native_edges(
    monkeypatch, kind, width, legacy
):
    import flagquantum.compiler.registers.modular_declarations as module
    from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    original = module.source_modular_arithmetic_resource_formula(
        kind, width, legacy_shims=legacy
    )
    seen = []

    def recipe(root):
        seen.append(root)
        return {FixedTCall("fredkin"): 5 * width}

    monkeypatch.setattr(module, "arithmetic_callees", recipe)
    actual = module.source_modular_arithmetic_resource_formula(
        kind, width, legacy_shims=legacy
    )
    assert seen == [ArithmeticCall(kind, (width,))]
    assert actual.cswap == 5 * width
    assert (
        actual.and_bloq == actual.toffoli == actual.clifford == actual.measurement == 0
    )
    assert actual.logical_qubits == original.logical_qubits
