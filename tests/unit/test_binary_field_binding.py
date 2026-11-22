"""Exercise original Python scalar semantics at native owned boundaries."""

import json
from pathlib import Path

import pytest

from flagquantum.core._binary_field import BinaryField
from flagquantum.ecosystem.binary_fields import (
    binary_field_value,
    binary_polynomial_value,
    resolve_binary_field,
)
from flagquantum.simulation.binary_field_scalar import (
    binary_field_operand,
    binary_field_power,
    binary_field_sum,
)
from tests.unit.fixtures.generate_binary_field_binding import external_value

pytestmark = pytest.mark.unit


def test_complete_original_python_scalar_observations() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("binary_field_binding.json")
        .read_text()
    )
    assert fixture["source_blob"] == "5f111727a1e383394ef856e7104ff4817ae12f31"
    for case in fixture["cases"]:
        operands = [external_value(value) for value in case["args"]]
        try:
            if case["kind"] == "construct":
                field = binary_field_value(*operands)
                result: list[int] | int = [
                    field.degree,
                    field.modulus,
                    field.primitive_element,
                ]
            else:
                field = BinaryField(case["degree"])
                operation = (
                    binary_field_sum if case["kind"] == "add" else binary_field_power
                )
                result = operation(field, *operands)
        except (ValueError, TypeError) as error:
            assert type(error).__name__ == case.get("error"), case
        else:
            assert "error" not in case and result == case["result"], case


def test_field_resolver_preserves_explicit_identity() -> None:
    field = binary_field_value(8, 0x11B, 3)
    assert resolve_binary_field(field, 8) is field
    assert resolve_binary_field(None, 8) == BinaryField(8)
    foreign = external_value(
        {"kind": "field", "char": 2, "degree": 8, "modulus": 0x11B, "primitive": 5}
    )
    assert resolve_binary_field(foreign, 8) == BinaryField(8, 0x11B, 5)
    for invalid in (field, foreign):
        with pytest.raises(ValueError, match="register size"):
            resolve_binary_field(invalid, 4)
    for invalid in (0x11B, "0x11B", True, object()):
        with pytest.raises(ValueError, match="Field requires"):
            resolve_binary_field(invalid, 8)


def test_conversion_consumes_external_values_and_validates_before_coercion() -> None:
    class IndexValue:
        def __index__(self) -> int:
            return 0x11B

    class DivergentInteger(IndexValue):
        def __int__(self) -> int:
            return 0x11D

    class BooleanCharacteristic(IndexValue):
        characteristic = True

    class MissingCharacteristic(IndexValue):
        characteristic = None

    assert binary_polynomial_value(IndexValue()) == 0x11B
    assert binary_polynomial_value(DivergentInteger()) == 0x11D
    for value in (
        True,
        -1,
        1 << 1024,
        1.5,
        BooleanCharacteristic(),
        MissingCharacteristic(),
    ):
        with pytest.raises(ValueError):
            binary_polynomial_value(value)
    assert binary_field_operand(True) == 1
    for invalid_operand in (IndexValue(), "1", 1.0):
        with pytest.raises(TypeError):
            binary_field_operand(invalid_operand)


def test_signed_power_and_reduced_xor_retain_scalar_scope() -> None:
    field = BinaryField(8, 0x11B)
    assert field.add(0x100, 1) == 0x101
    assert binary_field_sum(field, 0x100, 1) == 0x1A
    assert binary_field_power(field, 0x53, -1) == 0xCA
    assert binary_field_power(field, 0, -1) == 0
    assert binary_field_power(field, 0, 0) == 1
    assert binary_field_power(field, 3, -(1 << 4096) - 1) == field.power(
        field.inverse(3), (1 << 4096) + 1
    )
    assert hash(field) == hash(BinaryField(8, 0x11B, 3))
    assert field != BinaryField(8, 0x11B, 5)
    with pytest.raises(ValueError):
        field.power(3, -1)
