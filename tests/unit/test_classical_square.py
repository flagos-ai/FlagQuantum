"""Original square value semantics, including checked erasure and errors."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_square import square_classical_values
from tests.unit.test_classical_product_accumulation import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        {},
        {"a": 3},
        {"a": 3, "result": 9},
        {"a": 3, "result": 8},
        {"a": np.int8(100), "result": np.int8(16)},
        {"a": np.uint8(200), "result": np.uint8(64)},
        {"a": np.array([1, 2]), "result": np.array([1, 4])},
        {"a": np.array([3]), "result": np.array([9])},
        {"a": sympy.Symbol("a"), "result": sympy.Symbol("a") ** 2},
        {"a": sympy.Integer(3), "result": sympy.Integer(9)},
    ],
)
def test_square_forward_extra_output_and_adjoint_assertion(values, uncompute):
    from qualtran.bloqs.arithmetic.multiplication import Square

    source = Square(8, uncompute=uncompute)
    try:
        expected = source.on_classical_vals(**values)
    except Exception as error:
        with pytest.raises(type(error)):
            square_classical_values(values, uncompute=uncompute)
        return
    actual = square_classical_values(values, uncompute=uncompute)
    assert tuple(actual) == tuple(expected)
    for name, expected_value in expected.items():
        same_value(actual[name], expected_value)
    assert actual["a"] is values["a"]
