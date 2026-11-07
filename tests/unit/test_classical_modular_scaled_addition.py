"""Source scaled modular accumulation retains raw arithmetic and control types."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_modular_scaled_addition import (
    controlled_modular_scaled_add_classical_values,
)
from tests.unit.test_classical_modular_addition import compare_value_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "left,right",
    [
        (1, 2),
        (-1, 9),
        (9, 3),
        (np.uint8(200), np.uint8(250)),
        (np.array([1, 2]), np.array([0, 1])),
        (sympy.Integer(2), sympy.Integer(1)),
    ],
)
@pytest.mark.parametrize(
    "constant,modulus",
    [
        (1, 3),
        (-3, 3),
        (sympy.Integer(1), 3),
        (sympy.Symbol("k"), sympy.Symbol("p", integer=True, positive=True)),
    ],
)
@pytest.mark.parametrize("control", [0, 1, 2, np.uint8(0), np.uint8(1)])
def test_original_scaled_modadd_values_types_errors_and_activation(
    left, right, constant, modulus, control
):
    from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd

    source = CtrlScaleModAdd(constant, modulus, 8)

    def original(c, x, y):
        return tuple(source.on_classical_vals(ctrl=c, x=x, y=y).values())

    def native(c, x, y):
        return controlled_modular_scaled_add_classical_values(
            c, x, y, constant, modulus
        )

    compare_value_calls(original, native, (control, left, right))


def test_scaled_modadd_inactive_bypasses_zero_modulus_and_preserves_aliases():
    left = np.array([99, -1])
    right = np.array([0, 1])
    zero, x, y = controlled_modular_scaled_add_classical_values(
        np.uint8(0), left, right, 2, 0
    )
    assert type(zero) is int and x is left and y is right
    with pytest.raises(AssertionError, match="Bad ctrl value"):
        controlled_modular_scaled_add_classical_values(2, left, right, 2, 0)
