"""Distinct original ModAddK eager and CModAddK control-first value evaluation."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_modular_constant_addition import (
    controlled_modular_add_constant_classical_values,
    modular_add_constant_classical_values,
)
from tests.unit.test_classical_modular_addition import compare_value_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

VALUES = [-1, 0, 2, 3, np.uint8(2), np.array([1]), np.array([0, 1]), sympy.Integer(2)]
CONSTANTS = [1, -3, sympy.Integer(1), sympy.Symbol("k", integer=True)]
MODULI = [3, sympy.Integer(3), sympy.Symbol("p", integer=True, positive=True)]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("constant", CONSTANTS)
@pytest.mark.parametrize("modulus", MODULI)
@pytest.mark.parametrize(
    "pattern,control", [((), None), ((1,), 0), ((1,), 1), ((0,), 0), ((1, 0), 2)]
)
def test_original_modaddk_eager_arithmetic_and_packed_controls(
    value, constant, modulus, pattern, control
):
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAddK

    source = ModAddK(4, modulus, constant, cvs=pattern)

    def original(x):
        return tuple(source.on_classical_vals(x=x, ctrl=control).values())

    def native(x):
        return tuple(
            modular_add_constant_classical_values(
                x, constant, modulus, control=control, control_values=pattern
            ).values()
        )

    compare_value_calls(original, native, (value,))


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("constant", CONSTANTS)
@pytest.mark.parametrize("modulus", MODULI)
@pytest.mark.parametrize("control", [0, 1, 2, np.uint8(0), np.uint8(1)])
def test_original_cmodaddk_active_domain_control_types_and_inactive_bypass(
    value, constant, modulus, control
):
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK

    source = CModAddK(constant, modulus, 4)

    def original(c, x):
        return tuple(source.on_classical_vals(ctrl=c, x=x).values())

    def native(c, x):
        return controlled_modular_add_constant_classical_values(c, x, constant, modulus)

    compare_value_calls(original, native, (control, value))


@pytest.mark.parametrize(
    "pattern,control", [((1,), None), ((), 0), ((2,), 1), (1, 1), ((0, 1), 1)]
)
def test_pattern_conversion_and_original_assertion_order(pattern, control):
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAddK

    source = ModAddK(4, 3, 1, cvs=pattern)

    def original(x):
        return tuple(source.on_classical_vals(x=x, ctrl=control).values())

    def native(x):
        return tuple(
            modular_add_constant_classical_values(
                x, 1, 3, control=control, control_values=pattern
            ).values()
        )

    compare_value_calls(original, native, (1,))


def test_eager_and_control_first_are_not_interchangeable():
    values = np.array([0, 1])
    with pytest.raises(ValueError):
        modular_add_constant_classical_values(
            values, 1, 3, control=0, control_values=(1,)
        )
    zero, unchanged = controlled_modular_add_constant_classical_values(
        np.uint8(0), values, 1, 3
    )
    assert type(zero) is int and zero == 0 and unchanged is values
