"""Complete scalar source correspondence, storage ownership and operator routes."""

import shlex
from pathlib import Path
from typing import Literal

import pytest

from flagquantum.core._fixed_width_integer import FixedWidthInteger as F
from flagquantum.core._fixed_width_modular import (
    add_modulo,
    double_modulo,
    halve_modulo,
    inverse_modulo,
    is_coprime,
    negate_modulo,
    remainder_into,
)
from flagquantum.simulation.fixed_width_sampling import (
    sample_fixed_width,
    sample_fixed_width_modulo,
)
from flagquantum.simulation.physical_basis_rng import _mt19937_64
from tests.unit.fixtures.fixed_width_reference import IntegerCase, read_cases

pytestmark = pytest.mark.unit


def observe(case: IntegerCase) -> list[int] | list[float] | list[str]:
    op, w, x, v, y, t, z, scalar, shift, alias = case
    if op == "parse":
        radix: Literal[2, 10, 16] | None = None
        if scalar == 2:
            radix = 2
        elif scalar == 10:
            radix = 10
        elif scalar == 16:
            radix = 16
        parsed = F.from_text(x, w if alias else None, radix=radix)
        return [parsed.width, parsed.value, 0]
    if op.startswith("random"):
        draws = _mt19937_64(scalar)
        sampled = (
            sample_fixed_width(w, draws)
            if op == "random"
            else sample_fixed_width_modulo(
                w, F(v, int(y, 16)), draws, coprime=op == "random_coprime"
            )
        )
        return [sampled.width, sampled.value, 0, next(draws)]
    a, b, c = F(w, int(x, 16)), F(v, int(y, 16)), F(t, int(z, 16))
    other = a if alias == 1 else b
    third = a if alias == 2 else c
    flag = False
    if op.startswith("perf_"):
        for _ in range(shift):
            if op == "perf_scalar":
                a = a.multiply_into(scalar)
            elif op == "perf_product":
                a = a.accumulate_product(other, third)
            elif op == "perf_product_scalar":
                a = a.accumulate_product(scalar, other)
            elif op == "perf_remainder":
                a = remainder_into(a.xor_word(third), other)
            else:
                raise ValueError(op)
        return [a.width, a.value, 0]
    signed = scalar if scalar < 1 << 63 else scalar - (1 << 64)
    if op == "add":
        a, flag = a.add_word(other, bool(scalar))
    elif op == "sub":
        a, flag = a.subtract_word(other)
    elif op == "scalar_carry":
        a, flag = a.add_scalar_carry(scalar, bool(shift))
    elif op == "scalar_add":
        a = a.add_scalar(scalar)
    elif op == "scalar_sub":
        a = a.add_scalar(scalar, subtract=True)
    elif op == "signed_add":
        a = a.add_scalar(signed, signed=True)
    elif op == "signed_sub":
        a = a.add_scalar(signed, signed=True, subtract=True)
    elif op == "increment":
        a, flag = a.increment(scalar)
    elif op == "decrement":
        a, flag = a.decrement(scalar)
    elif op == "clip":
        a, flag = a.clip()
    elif op == "write":
        a = a.write_from(other)
    elif op == "and":
        a = a.and_word(other)
    elif op == "or":
        a = a.or_word(other)
    elif op == "xor":
        a = a.xor_word(other)
    elif op == "xor_int":
        a = a.xor_word(
            (scalar & ((1 << 32) - 1)) - (1 << 32)
            if scalar & (1 << 31)
            else scalar & ((1 << 32) - 1)
        )
    elif op == "left":
        a = a.shift_left(shift)
    elif op == "unsigned_left":
        a = a.shift_left(scalar)
    elif op == "unsigned_right":
        a = a.shift_right(scalar)
    elif op == "unsigned_read":
        a = F(64, a.read_shifted_word(scalar))
    elif op == "right":
        a = a.shift_right(shift)
    elif op == "full_left":
        a = a << shift
    elif op == "full_right":
        a = a >> shift
    elif op == "signed_right":
        a = a.shift_right(shift, signed=True)
    elif op == "add_shifted":
        a = a.add_shifted(other, shift)
    elif op == "sub_shifted":
        a, flag = a.subtract_shifted(other, shift)
    elif op == "multiply":
        a = a.multiply_into(other)
    elif op == "multiply_scalar":
        a = a.multiply_into(scalar)
    elif op == "product":
        a = a.accumulate_product(other, third)
    elif op == "product_scalar":
        a = a.accumulate_product(scalar, other)
    elif op == "product_shifted":
        a = a.accumulate_product(scalar, other, shift=shift)
    elif op == "subtract_product":
        a = a.accumulate_product(scalar, other, subtract=True)
    elif op == "subtract_product_shifted":
        a = a.accumulate_product(scalar, other, shift=shift, subtract=True)
    elif op == "times5":
        a, flag = a.times_five()
    elif op == "div5":
        a = a.divide_five()
    elif op == "negate":
        a = a.negate_into()
    elif op == "sum":
        a = a + other
    elif op == "difference":
        a = a - other
    elif op == "negative":
        a = -a
    elif op == "full_product":
        a = a * other
    elif op == "remainder":
        a = remainder_into(a, other)
    elif op == "mod":
        a = a % other
    elif op == "mod_add":
        a = add_modulo(a, other, third)
    elif op == "mod_sub":
        a = add_modulo(a, other, third, subtract=True)
    elif op == "mod_double":
        a = double_modulo(a, other)
    elif op == "mod_half":
        a = halve_modulo(a, other)
    elif op == "mod_negate":
        a = negate_modulo(a, other)
    elif op == "inverse":
        a = inverse_modulo(a, other)
    elif op == "coprime":
        flag = is_coprime(a, other)
    elif op == "read":
        a = F(64, a.read_shifted_word(shift))
    elif op == "shrink":
        a = a.resize_shrink(scalar)
    elif op == "clear":
        a = a.resize_clear(scalar)
    elif op == "set":
        a = a.set_bit(scalar)
    elif op == "reset":
        a = a.set_bit(scalar, False)
    elif op == "flip":
        a = a.flip_bit(scalar)
    elif op == "assign":
        a = a.assign_integer(scalar)
    elif op == "assign_signed":
        a = a.assign_integer(signed, signed=True)
    elif op == "ctor":
        a = F.from_integer(w, scalar)
    elif op == "ctor_signed":
        a = F.from_integer(w, signed, signed=True)
    elif op == "format":
        return [a.binary(), a.hexadecimal(), str(a), a.decimal()]
    elif op == "float":
        return [float(a), a.approximate_mantissa()]
    elif op == "compare":
        return list(
            map(
                int,
                [
                    a == other,
                    a < other,
                    a <= other,
                    a == signed,
                    a < signed,
                    a <= signed,
                ],
            )
        )
    else:
        raise ValueError(op)
    return [a.width, a.value, int(flag)]


def test_complete_unchanged_source_correspondence() -> None:
    count = 0
    fixture = (
        Path(__file__).with_name("data") / "fixed_width_integer_reference.jsonl.xz"
    )
    for case, line in read_cases(fixture):
        expected: str | list[int] | list[float] | list[str]
        if line == "REJECTED":
            expected = line
        elif case.operation == "format":
            expected = shlex.split(line)
        elif case.operation == "float":
            expected = list(map(float.fromhex, line.split()))
        elif case.operation == "compare":
            expected = list(map(int, line.split()))
        else:
            width, raw, *flags = line.split()
            expected = [int(width), int(raw, 16), *map(int, flags)]
        try:
            observed: str | list[int] | list[float] | list[str] = observe(case)
        except ValueError:
            observed = "REJECTED"
        assert observed == expected, case
        count += 1
    assert count == 74043


def test_python_operators_preserve_source_width_and_alias_rules() -> None:
    value = F(8, 7)
    value *= value
    assert (value.width, value.value) == (8, 76)
    value += -1
    assert (value.width, value.value) == (8, 75)
    value -= -1
    assert value.value == 76
    value <<= 1
    value >>= 1
    value ^= 7
    value &= F(8, 127)
    value |= F(8, 128)
    assert (value.width, value.value) == (8, 203)
    assert (1 + F(8, 7)).width == 73
    assert (1 - F(8, 7)).value == (1 << 74) - 6
    assert (2 * F(8, 7)).value == 14
    assert list(F(3, 5)) == [True, False, True]


def test_invalid_source_memory_and_nontermination_boundaries() -> None:
    empty = F()
    with pytest.raises(ValueError, match="allocated storage"):
        empty.assign_integer(1)
    with pytest.raises(ValueError, match="allocated storage"):
        empty.add_scalar_carry(1)
    value = F(7, 5)
    with pytest.raises(ValueError, match="cannot terminate"):
        remainder_into(value, value)
    with pytest.raises(ValueError, match="declared width"):
        value.reference_bit(7)
    with pytest.raises(ValueError, match="32-bit"):
        value.xor_word(1 << 32)
    assert value.write_from(value).value == 0
    assert value % value == 0
    other = F(128, 75)
    other %= F(7, 7)
    assert other.width == 128 and other.value == 5


def test_storage_views_and_declared_bits() -> None:
    value = F(65).set_bit(100).set_bit(0)
    assert value.reference_bit(0)
    assert not value.reference_bit(64)
    assert value.bit(100) and not value.bit(128)
    assert value.word_count == 2 and value.bits_in_use == 101
    assert value.clip() == (F(65, 1), True)
    assert value.flip_bit(100) == F(65, 1)
    assert value.resize_clear(7) == F(7)
    assert hash(F(7, 5)) == hash(F(65, 5))
    assert int(F(128, 7)) == 7
    with pytest.raises(ValueError, match="64-bit"):
        int(F(128, 1 << 65))
    with pytest.raises(ValueError, match="exponent range"):
        float(F(1025, 1 << 1024))
