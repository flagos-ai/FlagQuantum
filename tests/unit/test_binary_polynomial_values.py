"""Compare binary polynomial value semantics with unchanged source observations."""

import json
from pathlib import Path

import pytest

from flagquantum.core import _binary_polynomial as polynomial

pytestmark = pytest.mark.unit
_FIXTURE = Path(__file__).with_name("fixtures") / "binary_polynomial_values.json"
_CAPACITY = 1024
_MASK = (1 << _CAPACITY) - 1


def _observe(operation: str, left: str, right: str, amount: int) -> str:
    value, other = polynomial.parse(left), polynomial.parse(right)
    if operation == "mul":
        value = polynomial.multiply(value, other)
    elif operation == "mod":
        value = polynomial.remainder(value, other)
    elif operation == "gcd":
        value = polynomial.gcd(value, other)
    elif operation == "inv":
        value = polynomial.invert_mod(value, other)
    elif operation == "left":
        value = (value << amount) & _MASK
    elif operation == "right":
        value >>= amount
    elif operation == "xor":
        value ^= other
    elif operation in ("xorshift", "self"):
        value = polynomial.xor_shifted(
            value, value if operation == "self" else other, amount
        )
    elif operation in ("truncate", "fixed"):
        if operation == "fixed" and amount > _CAPACITY:
            raise ValueError("Fixed-width conversion exceeds capacity")
        value = polynomial.truncate(value, amount)
    elif operation in ("set", "clear", "toggle", "monomial"):
        if amount >= _CAPACITY:
            raise ValueError("Coefficient exceeds capacity")
        if operation == "set":
            value |= 1 << amount
        elif operation == "clear":
            value &= ~(1 << amount)
        elif operation == "toggle":
            value ^= 1 << amount
        else:
            value = 1 << amount
    elif operation == "algebraic":
        return polynomial.algebraic(value, amount)
    elif operation == "bit":
        return str((value >> amount) & 1)
    elif operation == "meta":
        # Native zero degree is represented by bit_length()-1. The unsigned
        # C++ sentinel belongs only to the observation conversion boundary.
        degree = value.bit_length() - 1 if value else (1 << 64) - 1
        return f"{int(value == 0)},{degree},{value.bit_length()},{value.bit_count()}"
    return f"0x{value:X}"


def test_complete_source_value_operations_and_refusals() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    assert {source["parent"] for source in fixture["sources"]} == {
        "K-0309",
        "K-0310",
        "K-0311",
    }
    for case in fixture["cases"]:
        arguments = (case["operation"], case["left"], case["right"], case["amount"])
        if case["expected"] == "ERROR":
            with pytest.raises(ValueError):
                _observe(*arguments)
        else:
            assert _observe(*arguments) == case["expected"], arguments


def test_shared_reduction_preserves_unbounded_compiler_inputs() -> None:
    from flagquantum.compiler.registers.finite_field.field_multiplication import (
        _remainder,
    )

    assert _remainder is polynomial.remainder
    assert _remainder(1 << 4096, 3) == 1
    for value in range(256):
        inverse = polynomial.invert_mod(value, 0x11B)
        assert polynomial.remainder(polynomial.multiply(value, inverse), 0x11B) == bool(
            value
        )


@pytest.mark.parametrize("value,width", [(-1, 1), (1, -1)])
def test_negative_storage_values_are_refused(value: int, width: int) -> None:
    with pytest.raises(ValueError):
        polynomial.truncate(value, width)


@pytest.mark.parametrize("left,right", [(-1, 0), (1, -1)])
def test_negative_polynomial_operands_are_refused(left: int, right: int) -> None:
    with pytest.raises(ValueError):
        polynomial.multiply(left, right)
    with pytest.raises(ValueError):
        polynomial.gcd(left, right)


def test_zero_modulus_and_negative_shift_fail_before_iteration() -> None:
    with pytest.raises(ValueError):
        polynomial.remainder(1, 0)
    with pytest.raises(ValueError):
        polynomial.invert_mod(1, 0)
    with pytest.raises(ValueError):
        polynomial.xor_shifted(1, 2, -1)
