"""Compare the complete modular scalar source with immutable native snapshots."""

import shlex
from collections import Counter
from pathlib import Path

import pytest

from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.core._modular_integer import ModularInteger
from flagquantum.simulation.fixed_width_sampling import sample_modular_integer
from flagquantum.simulation.physical_basis_rng import _mt19937_64

from .fixtures.modular_integer_reference import ModularCase, read_cases

pytestmark = pytest.mark.unit


def execute(case: ModularCase) -> tuple[ModularInteger | bool, int]:
    modulus = FixedWidthInteger(case.modulus_width, int(case.modulus, 16))
    value = ModularInteger(FixedWidthInteger(case.width, int(case.value, 16)), modulus)
    other = ModularInteger(
        FixedWidthInteger(case.other_width, int(case.other, 16)),
        FixedWidthInteger(case.other_width, int(case.other_modulus, 16)),
    )
    operation = case.operation
    if operation == "zero":
        return ModularInteger.zero(modulus), 0
    if operation == "signed":
        return ModularInteger.from_signed(case.scalar, modulus), 0
    if operation == "text":
        return ModularInteger.from_text(case.text, modulus), 0
    if operation in ("word", "word_move"):
        return (
            ModularInteger.from_word(
                value.value, modulus, moved=operation == "word_move"
            ),
            0,
        )
    if operation in ("copy", "move", "inspect"):
        return value, 0
    if operation in ("assign", "assign_move"):
        return other, 0
    if operation == "equal":
        return value == other, 0
    if operation == "add":
        return value.add(other), 0
    if operation == "add_scalar":
        return value.add(case.scalar), 0
    if operation == "sub":
        return value.subtract(other), 0
    if operation == "mul":
        return value.multiply(other), 0
    if operation == "mul_self":
        return value.multiply(value), 0
    if operation == "mul_scalar":
        return value.multiply(case.scalar), 0
    if operation == "square":
        return value.squared(), 0
    if operation in ("divide", "divide_move"):
        return value.divide(other, moved=operation == "divide_move"), 0
    if operation == "divide_scalar":
        return value.divide(case.scalar), 0
    if operation == "divide_self":
        return value.divide(value), 0
    if operation == "divide_self_move":
        return value.divide(value, moved=True), 0
    if operation == "negate":
        return value.negate(), 0
    if operation == "increment":
        return value.increment(), 0
    if operation == "inverse":
        return value.inverse(), 0
    if operation == "double":
        return value.double(), 0
    if operation == "halve":
        return value.halve(), 0
    if operation.startswith("left"):
        return value.shift_left(case.scalar, signed="_u" not in operation), 0
    if operation.startswith("right"):
        return value.shift_right(case.scalar, signed="_u" not in operation), 0
    if operation in ("random", "randomize"):
        draws = _mt19937_64(case.seed)
        result = sample_modular_integer(modulus, draws)
        if operation == "randomize":
            result = sample_modular_integer(modulus, draws)
        return result, next(draws)
    raise ValueError(f"Unknown reference operation {operation}")


def test_complete_original_modular_source_stream() -> None:
    counts: Counter[str] = Counter()
    fixture = Path(__file__).with_name("data") / "modular_integer.jsonl.xz"
    for case, expected in read_cases(fixture):
        fields = shlex.split(expected)
        if fields[0] == "equal":
            assert execute(case) == (fields[1] == "1", 0), case
            counts["equal"] += 1
            continue
        if fields[0] == "1":
            with pytest.raises(ValueError):
                execute(case)
            counts["failure"] += 1
            continue
        result, next_word = execute(case)
        assert isinstance(result, ModularInteger)
        assert result.value.width == int(fields[1]), case
        assert result.value.value == int(fields[2], 16), case
        assert bool(result) == (fields[4] == "1"), case
        assert next_word == int(fields[5]), case
        assert str(result) == fields[6], case
        if fields[3] == "1":
            result.verify_invariants()
        else:
            with pytest.raises(ValueError):
                result.verify_invariants()
        counts["success"] += 1
    assert counts == {"success": 10516, "failure": 1781, "equal": 145}


def test_modulus_identity_and_failure_snapshot_ownership() -> None:
    modulus = FixedWidthInteger(4, 15)
    original = ModularInteger.from_signed(7, modulus)
    copy = ModularInteger(original.value, original.modulus)
    successor = original.add(copy)
    assert successor.modulus is modulus
    assert original.value.value == copy.value.value == 7
    divisor = ModularInteger.from_signed(3, modulus)
    with pytest.raises(ValueError, match="coprime"):
        original.divide(divisor)
    assert original.value.value == 7 and divisor.value.value == 3
    assert original.divide(original).value.value == 1
    assert hash(original) == hash(copy)
    assert original != object()
    assert ModularInteger.from_signed(
        1, FixedWidthInteger(3, 7)
    ) == ModularInteger.from_signed(1, FixedWidthInteger(4, 11))


def test_original_scalar_constructor_and_addition_quirks() -> None:
    modulus = FixedWidthInteger(3, 7)
    assert ModularInteger.from_signed(10, modulus).value.value == 2
    assert ModularInteger.from_text("10", modulus).value.value == 3
    assert ModularInteger.from_signed(-10, modulus).value.value == 5
    with pytest.raises(ValueError):
        ModularInteger.from_text("-10", modulus)
    wide = ModularInteger(
        FixedWidthInteger(64, (1 << 64) - 1), FixedWidthInteger(64, (1 << 64) - 1)
    )
    assert wide.add(1).value.value == 1
    with pytest.raises(ValueError, match="overflows"):
        wide.multiply(-(1 << 63))


def test_source_nonunit_failure_poststate_and_self_move_are_explicit() -> None:
    fixture = Path(__file__).with_name("data") / "modular_integer.jsonl.xz"
    source = next(
        output
        for case, output in read_cases(fixture)
        if case.operation == "divide"
        and case.width == 2
        and case.value == "0x1"
        and case.other == "0x0"
        and case.modulus == "0x3"
    )
    assert shlex.split(source) == [
        "1",
        "2",
        "0000000000000000",
        "1",
        "0",
        "0",
        "0 (mod 3)",
    ]
    value = ModularInteger.from_signed(1, FixedWidthInteger(2, 3))
    with pytest.raises(ValueError, match="coprime"):
        value.divide(ModularInteger.zero(value.modulus))
    assert value.value.value == 1
    moved = value.divide(value, moved=True)
    assert moved.value.width == 0 and not moved
    with pytest.raises(ValueError, match="widths"):
        moved.verify_invariants()


def test_invariant_order_and_invalid_widths() -> None:
    with pytest.raises(ValueError, match="fill"):
        ModularInteger(
            FixedWidthInteger(3, 7), FixedWidthInteger(4, 7)
        ).verify_invariants()
    with pytest.raises(ValueError, match="smaller"):
        ModularInteger.from_signed(7, FixedWidthInteger(3, 7)).verify_invariants()
    with pytest.raises(ValueError, match="widths"):
        ModularInteger(
            FixedWidthInteger(4, 1), FixedWidthInteger(3, 7)
        ).verify_invariants()
    assert not ModularInteger.zero(FixedWidthInteger(0))
    assert not ModularInteger.zero(FixedWidthInteger(3, 7))
    value = ModularInteger.from_signed(1, FixedWidthInteger(3, 7))
    with pytest.raises(ValueError, match="64-bit"):
        value.shift_left(1 << 64, signed=False)
    with pytest.raises(ValueError, match="64 bits"):
        value.shift_right(-(1 << 63) - 1)
