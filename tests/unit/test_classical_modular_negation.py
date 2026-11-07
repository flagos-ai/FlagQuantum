"""Source strict-interior modular negation and controlled short circuits."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_modular_negation import (
    controlled_modular_negate_classical_values,
    modular_negate_classical_value,
)
from tests.unit.test_classical_modular_addition import CONTROLS, compare_value_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        np.int8(2),
        np.uint8(2),
        np.array([1]),
        np.array([0, 1]),
        sympy.Integer(2),
        sympy.Symbol("x"),
    ],
)
@pytest.mark.parametrize(
    "modulus", [3, sympy.Integer(3), sympy.Symbol("p", integer=True, positive=True)]
)
@pytest.mark.parametrize("control,cv", CONTROLS)
def test_original_modular_negation_boundaries_types_and_activation(
    value, modulus, control, cv
):
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModNeg, ModNeg

    if control is None:
        source = ModNeg(QUInt(3), modulus)

        def original(x):
            return tuple(source.on_classical_vals(x=x).values())

        def native(x):
            return (modular_negate_classical_value(x, modulus),)

        compare_value_calls(original, native, (value,))
    else:
        source = CModNeg(QUInt(3), modulus, cv)

        def original(c, x):
            return tuple(source.on_classical_vals(ctrl=c, x=x).values())

        def native(c, x):
            return controlled_modular_negate_classical_values(
                c, x, modulus, control_value=cv
            )

        compare_value_calls(original, native, (control, value))


def test_negation_keeps_inactive_and_exterior_objects():
    array = np.array([99, -2])
    assert controlled_modular_negate_classical_values(0, array, 3)[1] is array
    for value in (-1, 0, 3, 9):
        assert modular_negate_classical_value(value, 3) is value
    assert modular_negate_classical_value(1, 3) == 2
