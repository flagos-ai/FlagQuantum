"""Source subtraction direction, conversion policies and all coherent branches."""

from itertools import product

import numpy as np
import pytest
import sympy
import torch

from flagquantum.compiler.registers.subtraction import subtract_from_ir, subtract_ir
from flagquantum.simulation.classical_subtraction import (
    subtract_classical_values,
    subtract_from_classical_values,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import registers

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,width,reverse",
    tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 2, 3), (False, True))),
)
def test_every_small_classical_pair_in_both_source_directions(name, width, reverse):
    import qualtran
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    dt = getattr(qualtran, name)(width)
    signed = name == "QInt"
    source = SubtractFrom(dt) if reverse else Subtract(dt)
    native = subtract_from_classical_values if reverse else subtract_classical_values
    domain = (
        range(-(1 << (width - 1)), 1 << (width - 1)) if signed else range(1 << width)
    )
    for a, b in product(domain, repeat=2):
        expected = source.on_classical_vals(a=a, b=b)
        result = native(a, b, width, signed=signed)
        assert result == (expected["a"], expected["b"])
        assert type(result[1]) is type(expected["b"])


@pytest.mark.parametrize(
    "signed,reverse,kind",
    tuple(
        product(
            (False, True), (False, True), ("int8", "uint8", "int64", "array1", "array2")
        )
    ),
)
def test_numpy_order_of_arithmetic_and_integer_cast_is_not_unified(
    signed, reverse, kind
):
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    source = (SubtractFrom if reverse else Subtract)((QInt if signed else QUInt)(4))
    native = subtract_from_classical_values if reverse else subtract_classical_values
    if kind.startswith("array"):
        a = np.array([1] if kind == "array1" else [1, 2], dtype=np.int64)
        b = np.array([3] if kind == "array1" else [3, 4], dtype=np.int64)
    else:
        a, b = getattr(np, kind)(1), getattr(np, kind)(3)
    try:
        expected = source.on_classical_vals(a=a, b=b)
    except (ValueError, TypeError, OverflowError) as exc:
        with pytest.raises(type(exc)):
            native(a, b, 4, signed=signed)
    else:
        actual = native(a, b, 4, signed=signed)
        assert actual[0] is expected["a"] is a
        assert type(actual[1]) is type(expected["b"]) and actual[1] == expected["b"]


@pytest.mark.parametrize(
    "width,signed,reverse",
    tuple(
        product(
            (sympy.Integer(3), sympy.Symbol("n", positive=True, integer=True)),
            (False, True),
            (False, True),
        )
    ),
)
def test_original_symbolic_literal_and_unresolved_width_failures(
    width, signed, reverse
):
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    source = (SubtractFrom if reverse else Subtract)((QInt if signed else QUInt)(width))
    native = subtract_from_classical_values if reverse else subtract_classical_values
    try:
        expected = source.on_classical_vals(a=1, b=2)["b"]
    except (ValueError, TypeError) as exc:
        with pytest.raises(type(exc)):
            native(1, 2, width, signed=signed)
    else:
        result = native(1, 2, width, signed=signed)[1]
        assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize(
    "m,n,signed,reverse",
    (
        (1, 1, False, False),
        (1, 2, False, False),
        (2, 3, False, False),
        (1, 2, True, False),
        (2, 3, True, False),
        (1, 1, True, True),
        (2, 2, True, True),
        (3, 3, False, True),
    ),
)
def test_complete_complex_value_models_and_all_prefix_carry_cleanup(
    m, n, signed, reverse
):
    a, b, prefix, carry, bits = registers(m, n)
    gates = (
        subtract_from_ir(a, b, carry, bits)
        if reverse
        else subtract_ir(a, b, prefix, carry, bits, signed=signed)
    )
    work = len(prefix) + len(carry)
    total = m + n + work + 1
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amps
    expected = torch.zeros_like(state)
    native = subtract_from_classical_values if reverse else subtract_classical_values
    for label in data.tolist():
        left = label >> (n + 1)
        right = (label >> 1) & ((1 << n) - 1)
        if signed:
            left -= (1 << m) if left >= 1 << (m - 1) else 0
            right -= (1 << n) if right >= 1 << (n - 1) else 0
        x, y = native(left, right, n, signed=signed)
        logical = ((int(x) % (1 << m)) << n) | (int(y) % (1 << n))
        expected[(logical << (work + 1)) | (label & 1)] = amps[label]
    for history in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )
