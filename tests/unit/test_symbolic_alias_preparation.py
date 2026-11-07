"""Verify table-free alias dimensions and symbolic resource counts."""

import pytest

from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
    alias_probability_bits,
)

pytestmark = pytest.mark.unit


def test_numeric_probability_width_uses_minimum() -> None:
    assert alias_probability_bits(4, 1) == 1
    assert alias_probability_bits(4, 1 / 32) == 3
    with pytest.raises(ValueError, match="positive"):
        alias_probability_bits(4, 0)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [2, 3, 4, 6, None])
@pytest.mark.parametrize("symbolic_precision", [True, False])
@pytest.mark.parametrize("legacy", [False, True])
def test_shape_and_cost_match_source(
    count: int | None, symbolic_precision: bool, legacy: bool
) -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    rows = (
        count if count is not None else sympy.Symbol("L", positive=True, integer=True)
    )
    eps = sympy.Symbol("e", positive=True) if symbolic_precision else 0.125
    plan = AliasPreparationShape.from_precision(rows, rows, eps * rows)
    source = StatePreparationAliasSampling.from_n_coeff(
        rows, rows, precision=eps * rows
    )
    assert sympy.simplify(plan.probability_bits - source.mu) == 0
    assert sympy.simplify(plan.selection_width - source.selection_bitsize) == 0
    junk = sum(reg.total_bits() for reg in source.junk_registers)
    assert sympy.simplify(plan.junk_width - junk) == 0
    formula = plan.resource_model(legacy_shims=legacy)
    for op in (source, source.adjoint()):
        expected = get_cost_value(op, QECGatesCost(legacy_shims=legacy))
        for name in (
            "and_bloq",
            "clifford",
            "measurement",
            "rotation",
            "cswap",
            "toffoli",
            "t",
        ):
            assert sympy.simplify(getattr(formula, name) - getattr(expected, name)) == 0
        assert (
            sympy.simplify(formula.logical_qubits - get_cost_value(op, QubitCount()))
            == 0
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("domain", "support", "symbolic_precision"),
    [
        (7, 3, True),
        (10, 2, False),
        (19, 5, False),
        (7, None, True),
        (None, 3, True),
        (None, None, False),
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_sparse_shape_and_cost_match_source(
    domain: int | None, support: int | None, symbolic_precision: bool, legacy: bool
) -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    selection = (
        domain if domain is not None else sympy.Symbol("L", positive=True, integer=True)
    )
    rows = (
        support
        if support is not None
        else sympy.Symbol("d", positive=True, integer=True)
    )
    precision = sympy.Symbol("e", positive=True) if symbolic_precision else 0.03125
    plan = AliasPreparationShape.from_sparse_precision(selection, rows, 3, precision)
    source = SparseStatePreparationAliasSampling.from_n_coeff(
        selection, rows, 3, precision=precision
    )
    assert sympy.simplify(plan.probability_bits - source.mu) == 0
    assert sympy.simplify(plan.selection_width - source.selection_bitsize) == 0
    assert sympy.simplify(plan.sparse_index_width - source.sparse_index_bitsize) == 0
    assert (
        sympy.simplify(
            plan.junk_width - sum(r.total_bits() for r in source.junk_registers)
        )
        == 0
    )
    formula = plan.resource_model(legacy_shims=legacy)
    for op in (source, source.adjoint()):
        expected = get_cost_value(op, QECGatesCost(legacy_shims=legacy))
        for name in (
            "and_bloq",
            "clifford",
            "measurement",
            "rotation",
            "cswap",
            "toffoli",
            "t",
        ):
            assert sympy.simplify(getattr(formula, name) - getattr(expected, name)) == 0
        assert (
            sympy.simplify(formula.logical_qubits - get_cost_value(op, QubitCount()))
            == 0
        )


def test_sparse_shape_uses_full_domain_for_probability_precision() -> None:
    plan = AliasPreparationShape.from_sparse_precision(16, 2, 3, 1 / 32)
    assert (plan.selection_width, plan.sparse_index_width, plan.probability_bits) == (
        4,
        1,
        1,
    )
    assert plan.junk_width == 8
    singleton = AliasPreparationShape.from_sparse_precision(16, 1, 3, 1 / 32)
    assert singleton.sparse_index_width == 0
    with pytest.raises(ValueError, match="at least two"):
        singleton.resource_model()


@pytest.mark.parametrize("value", [None, 0, 1, "legacy"])
def test_symbolic_alias_profiles_require_literal_boolean(value):
    plan = AliasPreparationShape.from_precision(4, 4, 0.5)
    with pytest.raises(ValueError):
        plan.resource_model(legacy_shims=value)


def test_native_symbolic_alias_preserves_shared_model_replacement_and_profile(
    monkeypatch,
):
    from dataclasses import replace

    import flagquantum.algorithms.primitives.symbolic_alias_preparation as module

    plan = AliasPreparationShape.from_precision(5, 5, 0.5)
    original_qrom = module.source_qrom_resource_formula
    baseline = plan.resource_model(legacy_shims=True)
    observed = []

    def replacement(*args, **kwargs):
        observed.append(kwargs["legacy_shims"])
        model = original_qrom(*args, **kwargs)
        return replace(model, clifford=model.clifford + 7)

    monkeypatch.setattr(module, "source_qrom_resource_formula", replacement)
    changed = plan.resource_model(legacy_shims=True)
    assert observed == [True]
    assert changed.clifford == baseline.clifford + 7
    assert changed.and_bloq == baseline.and_bloq
    assert changed.logical_qubits == baseline.logical_qubits


def test_symbolic_model_queries_do_not_invoke_emitters(monkeypatch):
    import flagquantum.algorithms.primitives.alias_preparation as alias
    import flagquantum.compiler.registers.uniform_preparation as uniform

    def forbidden(*args, **kwargs):
        raise AssertionError("Shape model invoked an emitter")

    monkeypatch.setattr(uniform, "uniform_preparation_ir", forbidden)
    monkeypatch.setattr(alias, "alias_preparation_ir", forbidden)
    plan = AliasPreparationShape.from_sparse_precision(16, 3, 4, 1 / 32)
    modern, legacy = plan.resource_model(), plan.resource_model(legacy_shims=True)
    assert modern.and_bloq == legacy.and_bloq
    assert modern.logical_qubits == legacy.logical_qubits
    assert modern.clifford < legacy.clifford


def test_runnable_alias_resource_profile_example(capsys):
    from examples.algorithms.alias_resource_profiles import main

    main()
    assert "No circuit emission" in capsys.readouterr().out


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("legacy", [False, True])
def test_source_symbolic_substitution_and_concrete_rebuild_are_not_conflated(legacy):
    import sympy
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    rows = sympy.Symbol("L", positive=True, integer=True)
    symbolic = AliasPreparationShape.from_precision(rows, rows, 0.125).resource_model(
        legacy_shims=legacy
    )
    concrete = AliasPreparationShape.from_precision(5, 5, 0.125).resource_model(
        legacy_shims=legacy
    )
    source_symbolic = StatePreparationAliasSampling.from_n_coeff(
        rows, rows, precision=0.125
    )
    source_concrete = StatePreparationAliasSampling.from_n_coeff(5, 5, precision=0.125)
    expected_symbolic = get_cost_value(
        source_symbolic, QECGatesCost(legacy_shims=legacy)
    )
    expected_concrete = get_cost_value(
        source_concrete, QECGatesCost(legacy_shims=legacy)
    )
    assert (
        sympy.simplify(
            symbolic.clifford.subs({rows: 5})
            - concrete.clifford
            - (18 if legacy else 14)
        )
        == 0
    )
    assert (
        sympy.simplify(
            expected_symbolic.clifford.subs({rows: 5})
            - expected_concrete.clifford
            - (18 if legacy else 14)
        )
        == 0
    )
