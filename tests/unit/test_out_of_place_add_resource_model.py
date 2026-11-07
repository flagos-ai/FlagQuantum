"""Locked out-of-place addition declarations, categories and width dispatch."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
)
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_call_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = (
    1,
    3,
    8,
    sympy.Integer(0),
    sympy.Integer(3),
    sympy.Symbol("n", integer=True, positive=True),
    sympy.Symbol("p", integer=True, nonnegative=True),
    sympy.Symbol("z", integer=True),
)


@pytest.mark.parametrize(
    "width,msb,adjoint", tuple(product(WIDTHS, (False, True), (False, True)))
)
def test_original_direct_edges_and_recursive_qec_queries(width, msb, adjoint):
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    root = ArithmeticCall(
        "out_of_place_add", (width,), adjoint=adjoint, include_most_significant_bit=msb
    )
    source = OutOfPlaceAdder(
        width, is_adjoint=adjoint, include_most_significant_bit=msb
    )
    rows = arithmetic_call_graph(root)
    converted = {}
    for child, count in rows[root].items():
        if isinstance(child, ConjunctionCall):
            converted[And(uncompute=child.adjoint)] = count
        else:
            assert child == CliffordCall("cx", 2)
            converted[CNOT()] = count
    assert list(converted.items()) == list(source.build_call_graph(None).items())
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=rows, legacy_shims=legacy)
        expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )
