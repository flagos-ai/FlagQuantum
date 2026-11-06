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
def test_shape_and_cost_match_source(
    count: int | None, symbolic_precision: bool
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
    formula = plan.resource_model()
    for name, expected in get_cost_value(source, QECGatesCost()).asdict().items():
        assert sympy.simplify(getattr(formula, name) - expected) == 0
    assert (
        sympy.simplify(formula.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )
