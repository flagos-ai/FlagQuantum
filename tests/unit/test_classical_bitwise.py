"""Original classical values and complex-state bit-pattern consistency."""

from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.bits.bitwise import (
    bitwise_not_ir,
    xor_constant_ir,
    xor_ir,
)
from flagquantum.simulation.classical_bitwise import (
    bitwise_not_classical_value,
    xor_classical_values,
    xor_constant_classical_value,
)

pytestmark = pytest.mark.unit


def equal(actual, expected):
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    if isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "kind,scalar_type",
    tuple(
        product(
            ("xor", "constant", "not"), (int, np.int8, np.uint8, np.int64, np.uint64)
        )
    ),
)
def test_scalar_types_values_and_source_exceptions(kind, scalar_type):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK

    width = 4
    for number in (0, 1, 7, 15):
        x = scalar_type(number)
        if kind == "xor":
            expected = Xor(QUInt(width)).on_classical_vals(x=x, y=scalar_type(6))
            source, target = xor_classical_values(x, scalar_type(6))
            equal(source, expected["x"])
            equal(target, expected["y"])
        elif kind == "constant":
            for constant in (0, 5, 15, -1, 256):
                try:
                    expected = XorK(QUInt(width), constant).on_classical_vals(x=x)["x"]
                except (ValueError, OverflowError, TypeError) as exc:
                    with pytest.raises(type(exc)):
                        xor_constant_classical_value(x, constant)
                else:
                    equal(xor_constant_classical_value(x, constant), expected)
        else:
            try:
                expected = BitwiseNot(QUInt(width)).on_classical_vals(x=x)["x"]
            except (ValueError, OverflowError, TypeError) as exc:
                with pytest.raises(type(exc)):
                    bitwise_not_classical_value(x, width, unsigned=True)
            else:
                equal(bitwise_not_classical_value(x, width, unsigned=True), expected)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype", ("QAny", "QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("array_type", (np.int8, np.int64, np.uint8))
def test_array_broadcasting_and_input_ownership(dtype, array_type):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK

    dt = getattr(qualtran, dtype)(4)
    x = np.array([[0, 1, 5], [2, 7, 15]], dtype=array_type)
    y = np.array([1, 3, 6], dtype=array_type)
    old_x = x.copy()
    old_y = y.copy()
    expected = Xor(dt).on_classical_vals(x=x, y=y)
    source, target = xor_classical_values(x, y)
    assert source is x
    equal(target, expected["y"])
    equal(xor_constant_classical_value(x, 5), XorK(dt, 5).on_classical_vals(x=x)["x"])
    equal(
        bitwise_not_classical_value(
            x, 4, unsigned=dtype in ("QUInt", "QMontgomeryUInt")
        ),
        BitwiseNot(dt).on_classical_vals(x=x)["x"],
    )
    equal(x, old_x)
    equal(y, old_y)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype", ("QAny", "QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_every_classical_word_and_full_complex_ir_consistency(dtype, width):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, XorK

    dt = getattr(qualtran, dtype)(width)
    raw = torch.arange(1 << width, dtype=torch.int64)
    coeff = raw.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    values = [
        int(x) if dtype != "QInt" or x < 1 << (width - 1) else int(x) - (1 << width)
        for x in raw
    ]
    for kind in ("constant", "not"):
        constant = -1 if dtype == "QInt" else ((1 << width) - 1) // 3
        expected = []
        for x in values:
            if kind == "constant":
                value = xor_constant_classical_value(x, constant)
                assert value == XorK(dt, constant).on_classical_vals(x=x)["x"]
            else:
                value = bitwise_not_classical_value(
                    x, width, unsigned=dtype in ("QUInt", "QMontgomeryUInt")
                )
                assert value == BitwiseNot(dt).on_classical_vals(x=x)["x"]
            expected.append(int(value) % (1 << width))
        gates = (
            xor_constant_ir(tuple(range(width)), constant % (1 << width))
            if kind == "constant"
            else bitwise_not_ir(tuple(range(width)))
        )
        wanted = torch.zeros_like(amps)
        wanted[torch.tensor(expected)] = amps
        state = (
            Circuit(width, inputs=amps, dtype=torch.complex128)
            .append(FieldOperation(width, gates))
            .state()
            .flatten()
        )
        torch.testing.assert_close(state, wanted, rtol=0, atol=1e-12)


@pytest.mark.ftqc_reference
def test_symbolic_constants_remain_rejected_including_integer_literals():
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK

    for k in (sympy.Integer(5), sympy.Symbol("k", integer=True)):
        with pytest.raises(ValueError):
            XorK(QUInt(4), k).on_classical_vals(x=3)
        with pytest.raises(ValueError):
            xor_constant_classical_value(3, k)


@pytest.mark.parametrize(
    "width,unsigned", ((0, True), (-1, True), (True, True), (3, 1))
)
def test_invalid_width_and_encoding(width, unsigned):
    with pytest.raises(ValueError):
        bitwise_not_classical_value(0, width, unsigned=unsigned)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype", ("QAny", "QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("width", (1, 2, 3))
def test_xor_every_classical_pair_and_complete_complex_ir(dtype, width):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import Xor

    original = Xor(getattr(qualtran, dtype)(width))
    raw = torch.arange(1 << (2 * width + 1), dtype=torch.int64)
    coeff = raw.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    expected = torch.zeros_like(amplitudes)
    for label in raw.tolist():
        left = label >> (width + 1)
        right = (label >> 1) & ((1 << width) - 1)
        if dtype == "QInt":
            left -= (1 << width) if left >= 1 << (width - 1) else 0
            right -= (1 << width) if right >= 1 << (width - 1) else 0
        actual = xor_classical_values(left, right)
        source = original.on_classical_vals(x=left, y=right)
        assert actual == (source["x"], source["y"])
        mapped = (
            ((actual[0] % (1 << width)) << (width + 1))
            | ((actual[1] % (1 << width)) << 1)
            | (label & 1)
        )
        expected[mapped] = amplitudes[label]
    gates = xor_ir(tuple(range(width)), tuple(range(width, 2 * width)))
    assert len(gates) == width and all(g.name == "cx" for g in gates)
    state = (
        Circuit(2 * width + 1, inputs=amplitudes, dtype=torch.complex128)
        .append(FieldOperation(2 * width + 1, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(state, expected, rtol=0, atol=1e-12)
