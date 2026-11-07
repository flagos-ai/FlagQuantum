"""Original unequal and symbolic comparison declarations and native cleanup."""

from itertools import product

import pytest

from flagquantum.compiler.registers.comparison_resource_model import (
    source_less_equal_cost,
    source_less_equal_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = (
    (1, 1),
    (2, 2),
    (3, 3),
    (2, 3),
    (3, 2),
    (8, 8),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 5),
    (5, 1),
    (3, 8),
    (8, 3),
    (3, 6),
    (6, 3),
)


def assert_original(formula, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import simplify

    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for key in (
        "and_bloq",
        "clifford",
        "measurement",
        "rotation",
        "cswap",
        "toffoli",
        "t",
    ):
        assert simplify(getattr(formula, key) - getattr(expected, key)) == 0
    assert simplify(formula.logical_qubits - get_cost_value(source, QubitCount())) == 0


@pytest.mark.parametrize("widths,legacy", tuple(product(WIDTHS, (False, True))))
def test_concrete_source_tree_prefixes_and_widths(widths, legacy):
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual

    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    formula = source_less_equal_resource_formula(*widths, legacy_shims=legacy)
    source = LessThanEqual(*widths)
    assert_original(formula, source, legacy)
    assert_original(formula, source.adjoint(), legacy)
    model = estimate_ftqc_resources(
        source_less_equal_cost(*widths, legacy_shims=legacy)
    )
    assert model.emitted is None
    assert formula.clifford == model.logical.gates.clifford
    assert formula.logical_qubits == model.logical.peak_logical_qubits


@pytest.mark.parametrize("case,legacy", tuple(product(range(6), (False, True))))
def test_symbolic_widths_retain_original_left_only_declared_calls(case, legacy):
    import sympy
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual

    m, n = sympy.symbols("m n", positive=True, integer=True)
    widths = (
        (m, n),
        (m, 3),
        (2, n),
        (sympy.Integer(2), sympy.Integer(5)),
        (sympy.Integer(3), 3),
        (3, sympy.Integer(3)),
    )[case]
    formula = source_less_equal_resource_formula(*widths, legacy_shims=legacy)
    source = LessThanEqual(*widths)
    assert_original(formula, source, legacy)
    assert_original(formula, source.adjoint(), legacy)
    assert sympy.simplify(formula.and_bloq - (2 * widths[0] + 1)) == 0


@pytest.mark.parametrize("widths", [(0, 2), (2, 0), (True, 2), (2, 1.0)])
def test_invalid_widths_fail_before_a_resource_report(widths):
    with pytest.raises(ValueError):
        source_less_equal_resource_formula(*widths)


@pytest.mark.parametrize("flag", [None, 0, 1, "legacy"])
def test_profile_validation_applies_to_concrete_and_symbolic_branches(flag):
    from sympy import Symbol

    for widths in ((2, 5), (Symbol("n", positive=True, integer=True), 3)):
        with pytest.raises(ValueError):
            source_less_equal_resource_formula(*widths, legacy_shims=flag)


def test_comparison_cost_queries_never_invoke_an_emitter(monkeypatch):
    import flagquantum.compiler.registers.less_equal as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Cost query invoked comparison emitter")

    monkeypatch.setattr(module, "less_equal_ir", forbidden)
    assert source_less_equal_cost(2, 3).gates.measurement == 4
    assert source_less_equal_resource_formula(2, 3).and_bloq == 4


def test_runnable_unequal_width_model_and_execution_example(capsys):
    from examples.algorithms.register_less_equal import main

    main()
    output = capsys.readouterr().out
    assert "double XOR verified" in output
    assert "Emitted T/Tdag: 16" in output
