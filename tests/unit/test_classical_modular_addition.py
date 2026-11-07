"""Unmodified modular value methods retain boundaries, types and short circuits."""

from copy import deepcopy

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_modular_addition import (
    controlled_modular_add_classical_values,
    modular_add_classical_values,
)
from tests.unit.test_classical_product_accumulation import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

VALUES = [
    (1, 2),
    (3, 0),
    (0, 3),
    (3, 3),
    (-1, 1),
    (3, -1),
    (-1, 3),
    (9, 3),
    (np.int8(2), np.int8(1)),
    (np.uint8(2), np.uint8(0)),
    (np.array([1]), np.array([2])),
    (np.array([0, 1]), np.array([1, 2])),
    (sympy.Integer(1), sympy.Integer(2)),
]
MODULI = [0, 3, sympy.Integer(3), sympy.Symbol("p", integer=True, positive=True)]
CONTROLS = [(None, 1), (0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]


def compare_value_calls(source_call, native_call, values):
    original = deepcopy(values)
    native = deepcopy(values)
    try:
        expected = source_call(*original)
    except Exception as error:
        with pytest.raises(type(error)):
            native_call(*native)
    else:
        actual = native_call(*native)
        for a, e in zip(actual, expected, strict=True):
            same_value(a, e)
    # Include visible mutation, even when an intermediate operation fails.
    for a, e in zip(native, original, strict=True):
        same_value(a, e)


@pytest.mark.parametrize("values", VALUES)
@pytest.mark.parametrize("modulus", MODULI)
@pytest.mark.parametrize("control,cv", CONTROLS)
def test_original_modular_add_checks_order_and_inactive_identity(
    values, modulus, control, cv
):
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd

    if control is None:
        source = ModAdd(3, modulus)

        def source_call(a, b):
            return tuple(source.on_classical_vals(x=a, y=b).values())

        def native_call(a, b):
            return modular_add_classical_values(a, b, modulus)

        compare_value_calls(source_call, native_call, values)
    else:
        source = CModAdd(QUInt(3), modulus, cv)

        def source_call(c, a, b):
            return tuple(source.on_classical_vals(ctrl=c, x=a, y=b).values())

        def native_call(c, a, b):
            return controlled_modular_add_classical_values(
                c, a, b, modulus, control_value=cv
            )

        compare_value_calls(source_call, native_call, (control, *values))


def test_one_equal_modulus_exemption_keeps_original_diagnostic_scope():
    # The source exemption also admits an out-of-domain other operand. This is
    # value-method fidelity, not a clean quantum domain claim.
    assert modular_add_classical_values(-1, 3, 3) == (-1, 2)
    assert modular_add_classical_values(3, 9, 3) == (3, 0)
    with pytest.raises(ValueError):
        modular_add_classical_values(3, 3, 3)
    left, right = np.array([-1, 9]), np.array([-4, 7])
    result = controlled_modular_add_classical_values(0, left, right, 0)
    assert result[1] is left and result[2] is right
