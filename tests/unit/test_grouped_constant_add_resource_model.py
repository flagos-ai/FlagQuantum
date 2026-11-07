"""Original grouped-control symbolic AddK cost queries and source width fallback."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler.registers.grouped_constant_add_resource_model import (
    source_grouped_constant_add_resource_formula,
)
from tests.unit.test_grouped_controlled_constant_addition import EXTRA
from tests.unit.test_register_control_predicate import CASES, control_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n, k = sympy.symbols("n k", integer=True, positive=True)
z = sympy.Symbol("z", integer=True)
DIMENSIONS = (
    (3, 5),
    (3, sympy.Integer(5)),
    (sympy.Integer(3), 5),
    (n, k),
    (n, -1),
    (z, k),
    (n + 1, 0),
    (1, sympy.Integer(0)),
)


@pytest.mark.parametrize(
    "case,dimensions,adjoint,legacy",
    tuple(product(CASES + EXTRA, DIMENSIONS, (False, True), (False, True))),
)
def test_all_original_categories_and_fallback_with_unchanged_dimension_objects(
    case, dimensions, adjoint, legacy
):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    spec, registers, _ = control_fixture(case)
    width, constant = dimensions
    original = AddK(QUInt(width), constant).controlled(ctrl_spec=spec)
    if adjoint:
        original = original.adjoint()
    actual = source_grouped_constant_add_resource_formula(
        width, constant, registers, adjoint=adjoint, legacy_shims=legacy
    )
    expected = get_cost_value(original, QECGatesCost(legacy_shims=legacy))
    for field in fields(actual):
        value = (
            get_cost_value(original, QubitCount())
            if field.name == "logical_qubits"
            else getattr(expected, field.name)
        )
        assert sympy.simplify(getattr(actual, field.name) - value) == 0


def test_literal_reconstruction_keeps_inner_fallback_and_outer_live_predicates():
    from flagquantum.compiler.registers.control_predicate import ControlRegister

    regs = (ControlRegister(2, (1,)),)
    concrete = source_grouped_constant_add_resource_formula(3, 5, regs)
    constant_literal = source_grouped_constant_add_resource_formula(
        3, sympy.Integer(5), regs
    )
    width_literal = source_grouped_constant_add_resource_formula(
        sympy.Integer(3), 5, regs
    )
    symbolic = source_grouped_constant_add_resource_formula(n, k, regs)
    assert (
        concrete.logical_qubits,
        constant_literal.logical_qubits,
        width_literal.logical_qubits,
    ) == (11, 10, 8)
    assert symbolic.logical_qubits.subs(n, 3) == 8
    assert (
        concrete.clifford == 16
        and constant_literal.clifford == width_literal.clifford == 18
    )


def test_all_shared_child_categories_and_symbolic_width_connections_remain_replaceable(
    monkeypatch,
):
    import flagquantum.compiler.registers.grouped_constant_add_resource_model as module
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
    )
    from flagquantum.compiler.registers.control_predicate import ControlRegister
    from flagquantum.compiler.registers.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    addition = SymbolicResourceFormula(
        2, 3, 5, 29, t=7, toffoli=11, cswap=13, rotation=17
    )
    compute = ComponentCost("compute", 5, GateCost(t=2, and_bloq=3, clifford=7))
    erase = ComponentCost("erase", 7, GateCost(measurement=11, clifford=13))
    seen = []

    def positive(width, constant, **kwargs):
        assert width is n and constant is k
        seen.append(kwargs)
        return addition

    monkeypatch.setattr(
        module, "source_controlled_constant_add_resource_formula", positive
    )
    import flagquantum.compiler.registers.controlled_component_resource_model as shared

    monkeypatch.setattr(
        shared,
        "control_predicate_cost",
        lambda *args, uncompute=False, **kwargs: erase if uncompute else compute,
    )
    actual = module.source_grouped_constant_add_resource_formula(
        n, k, (ControlRegister(1, (1, 0, 1)),), adjoint=True, legacy_shims=True
    )
    assert seen == [dict(adjoint=True, legacy_shims=True)]
    assert actual == SymbolicResourceFormula(
        5, 23, 16, sympy.Max(33, n + 7), t=9, toffoli=11, cswap=13, rotation=17
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(registers=()),
        dict(adjoint=1),
        dict(legacy_shims=1),
        dict(width=True),
        dict(constant=True),
    ],
)
def test_symbolic_query_does_not_coerce_invalid_contract_inputs(kwargs):
    from flagquantum.compiler.registers.control_predicate import ControlRegister

    with pytest.raises(ValueError):
        source_grouped_constant_add_resource_formula(
            **(
                dict(width=n, constant=k, registers=(ControlRegister(2, (1,)),))
                | kwargs
            )
        )
