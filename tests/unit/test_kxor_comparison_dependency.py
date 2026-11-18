"""Close the concrete/symbolic comparator dependency of the k-XOR RHS graph."""

import pytest
import sympy
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.resource_counting import QECGatesCost, get_cost_value

from flagquantum.compiler.registers.comparison.comparison_resource_model import (
    source_less_than_constant_qec_cost,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,threshold", [(1, 0), (3, 0), (3, 1), (3, 7), (3, 8), (3, 9), (8, 3)]
)
def test_concrete_categories_reuse_existing_comparator(
    width: int, threshold: int
) -> None:
    expected = get_cost_value(LessThanConstant(width, threshold), QECGatesCost())
    actual = source_less_than_constant_qec_cost(width, threshold)
    for name in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def test_symbolic_and_partly_bound_source_graphs() -> None:
    width, threshold = sympy.symbols("width threshold", integer=True, positive=True)
    for n, k in (
        (width, threshold),
        (width, 3),
        (8, threshold),
        (sympy.Integer(8), sympy.Integer(3)),
    ):
        actual = source_less_than_constant_qec_cost(n, k)
        expected = get_cost_value(LessThanConstant(n, k), QECGatesCost())
        for name in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert sympy.simplify(getattr(actual, name) - getattr(expected, name)) == 0
