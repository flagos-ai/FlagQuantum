"""Original symbolic/concrete uniform costs and explicit source failure boundaries."""

from itertools import product

import pytest

from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = ((), (0,), (1,), (0, 1), (1, 0, 1))


@pytest.mark.parametrize(
    "case,values,legacy", tuple(product(range(9), PATTERNS, (False, True)))
)
def test_original_uniform_categories_and_width_fallbacks(case, values, legacy):
    import sympy
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = sympy.Symbol("L", positive=True, integer=True)
    count = (n, sympy.Integer(3), sympy.Integer(4), sympy.Integer(5), 2, 3, 4, 5, 6)[
        case
    ]
    formula = source_uniform_preparation_resource_formula(
        count, control_values=values, legacy_shims=legacy
    )
    source = PrepareUniformSuperposition(count, values)
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


@pytest.mark.parametrize("legacy", (False, True))
def test_symbolic_control_omission_is_retained_without_claiming_executable_control_parity(
    legacy,
):
    from sympy import Symbol, simplify

    n = Symbol("L", positive=True, integer=True)
    plain = source_uniform_preparation_resource_formula(n, legacy_shims=legacy)
    controlled = source_uniform_preparation_resource_formula(
        n, control_values=(1, 0, 1), legacy_shims=legacy
    )
    for name in ("and_bloq", "clifford", "measurement", "rotation"):
        assert simplify(getattr(plain, name) - getattr(controlled, name)) == 0
    assert plain.logical_qubits != controlled.logical_qubits


@pytest.mark.parametrize("values", PATTERNS)
def test_original_sympy_integer_two_peak_failure_is_not_replaced_by_guessed_width(
    values,
):
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value
    from sympy import Integer

    with pytest.raises(RuntimeError):
        get_cost_value(PrepareUniformSuperposition(Integer(2), values), QubitCount())
    with pytest.raises(ValueError, match="peak query"):
        source_uniform_preparation_resource_formula(Integer(2), control_values=values)


@pytest.mark.parametrize("count", (0, 1, True, 3.0))
def test_invalid_concrete_domain_and_singleton_reject(count):
    with pytest.raises(ValueError):
        source_uniform_preparation_resource_formula(count)


@pytest.mark.parametrize("value", [None, 0, 1, "legacy"])
def test_profiles_require_boolean_flags_in_both_branches(value):
    from sympy import Symbol

    for n in (5, Symbol("n", positive=True, integer=True)):
        with pytest.raises(ValueError):
            source_uniform_preparation_resource_formula(n, legacy_shims=value)


@pytest.mark.parametrize("pattern", [(False,), (2,), (-1,)])
def test_invalid_controls_cannot_hide_in_symbolic_omitted_calls(pattern):
    from sympy import Symbol

    with pytest.raises(ValueError):
        source_uniform_preparation_resource_formula(
            Symbol("n", positive=True, integer=True), control_values=pattern
        )


def test_symbolic_integer_does_not_dispatch_to_concrete_emitter_model(monkeypatch):
    import sympy

    import flagquantum.compiler.registers.preparation.uniform_preparation_resource_model as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Symbolic width was normalized into the concrete path")

    monkeypatch.setattr(module, "source_uniform_preparation_cost", forbidden)
    result = module.source_uniform_preparation_resource_formula(
        sympy.Integer(3), control_values=(0,)
    )
    assert result.logical_qubits == 3
    assert result.and_bloq == 5


def test_concrete_formula_consumes_shared_model_and_profile_replacement(monkeypatch):
    from dataclasses import replace

    import flagquantum.compiler.registers.preparation.uniform_preparation_resource_model as module

    original = module.source_uniform_preparation_cost
    baseline = module.source_uniform_preparation_resource_formula(
        6, control_values=(1, 0), legacy_shims=True
    )
    observed = []

    def replacement(count, **kwargs):
        observed.append((count, kwargs["control_values"], kwargs["legacy_shims"]))
        model = original(count, **kwargs)
        return replace(
            model, gates=replace(model.gates, clifford=model.gates.clifford + 7)
        )

    monkeypatch.setattr(module, "source_uniform_preparation_cost", replacement)
    changed = module.source_uniform_preparation_resource_formula(
        6, control_values=(1, 0), legacy_shims=True
    )
    assert observed == [(6, (1, 0), True)]
    assert changed.clifford == baseline.clifford + 7
    assert changed.logical_qubits == baseline.logical_qubits
