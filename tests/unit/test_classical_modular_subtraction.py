"""Source modular subtraction preserves in-place NumPy effects and no-op domains."""

import numpy as np
import pytest

from flagquantum.simulation.classical_modular_subtraction import (
    controlled_modular_subtract_classical_values,
    modular_subtract_classical_values,
)
from tests.unit.test_classical_modular_addition import (
    CONTROLS,
    MODULI,
    VALUES,
    compare_value_calls,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("values", VALUES)
@pytest.mark.parametrize("modulus", MODULI)
@pytest.mark.parametrize("control,cv", CONTROLS)
def test_original_modular_subtract_domain_types_mutation_and_errors(
    values, modulus, control, cv
):
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub, ModSub

    if control is None:
        source = ModSub(QUInt(3), modulus)

        def source_call(a, b):
            return tuple(source.on_classical_vals(x=a, y=b).values())

        def native_call(a, b):
            return modular_subtract_classical_values(a, b, modulus)

        compare_value_calls(source_call, native_call, values)
    else:
        source = CModSub(QUInt(3), modulus, cv)

        def source_call(c, a, b):
            return tuple(source.on_classical_vals(ctrl=c, x=a, y=b).values())

        def native_call(c, a, b):
            return controlled_modular_subtract_classical_values(
                c, a, b, modulus, control_value=cv
            )

        compare_value_calls(source_call, native_call, (control, *values))


def test_array_alias_mutation_and_unsigned_overflow_are_not_repaired():
    left, right = np.array([2]), np.array([1])
    original_right = right
    a, result = modular_subtract_classical_values(left, right, 3)
    assert a is left and result is original_right
    np.testing.assert_array_equal(right, [2])
    # Original uint8 -= underflows; the following negative test cannot fix it.
    _, result = modular_subtract_classical_values(np.uint8(2), np.uint8(0), 3)
    assert result == np.uint8(254)
