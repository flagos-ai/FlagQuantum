"""Compare exact phase words and rounding with portable native-source records."""

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from math import ldexp, pi, sin
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128 as Angle
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit

_REFERENCE = json.loads(
    (Path(__file__).with_name("data") / "phase_angle_128_reference.json").read_text()
)


def _evaluate(case: list[str]) -> Angle:
    op, *args = case
    if op == "p":
        return Angle.from_power_of_two_half_turns(int(args[0]))
    if op == "d":
        return Angle.from_half_turns_exact_double(float.fromhex(args[0]))
    if op == "r":
        return Angle.from_rounded_decimal_half_turns(args[0])
    if op == "i":
        return Angle.from_integer_half_turns(int(args[0]))
    a = Angle(int(args[0]), int(args[1]))
    if op == "w":
        return a
    if op == "n":
        return -a
    if op == "h":
        return a.rotated_half_turn()
    if op == "m":
        return a * int(args[2])
    if op == "j":
        return a.plus_integer_half_turns(int(args[2]))
    assert op == "a"
    return a + Angle(int(args[2]), int(args[3]))


@pytest.mark.parametrize(
    "record",
    _REFERENCE["records"],
    ids=[f"native-{i}" for i in range(len(_REFERENCE["records"]))],
)
def test_native_words_decimal_and_predicates(record: dict[str, Any]) -> None:
    assert _REFERENCE["source_commit"] == "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"
    assert _REFERENCE["source_blob"] == "492b3d63d9bc6678283c974c99cc3bcaf91d9d3c"
    if record["native"] == "E":
        with pytest.raises(ValueError):
            _evaluate(record["input"])
        return
    angle = _evaluate(record["input"])
    actual = "\t".join(
        map(
            str,
            (
                "V",
                angle.low,
                angle.high,
                angle.clifford_hierarchy_level(),
                int(angle.closer_to_half_turn_than_zero()),
                int(angle.is_multiple_of_45_degrees()),
                int(angle.is_multiple_of_90_degrees()),
                int(angle.is_multiple_of_180_degrees()),
                int(bool(angle)),
                angle.to_decimal_half_turns(),
            ),
        )
    )
    assert actual == record["native"]
    assert Angle.from_rounded_decimal_half_turns(angle.to_decimal_half_turns()) == angle
    assert str(angle) == angle.to_decimal_half_turns() + "π"
    signed = angle.half_turns()
    if signed >= 1:
        signed -= 2
    error = abs(Fraction.from_float(angle.signed_half_turns_float()) - signed)
    assert error <= Fraction(1, 1 << 54)


@pytest.mark.parametrize(
    "units", [0, 1, 2, (1 << 64) - 1, 1 << 64, (1 << 127) - 1, (1 << 128) - 1]
)
@pytest.mark.parametrize("negative", [False, True])
def test_exact_ties_and_sticky_digits_past_the_128_digit_prefix(
    units: int, negative: bool
) -> None:
    with localcontext() as context:
        context.prec = 300
        midpoint = Decimal(2 * units + 1) / Decimal(1 << 128)
        epsilon = Decimal(1) / (Decimal(10) ** 160)
        for delta, wanted in (
            (-epsilon, units),
            (Decimal(0), units + (units & 1)),
            (epsilon, units + 1),
        ):
            text = format(midpoint + delta, "f")
            if negative:
                text = "-" + text
            expected = Angle(wanted & ((1 << 64) - 1), wanted >> 64 & ((1 << 64) - 1))
            if negative:
                expected = -expected
            assert Angle.from_rounded_decimal_half_turns(text) == expected


@pytest.mark.parametrize(
    "text", ["", "0", "1.", ".5", "1e-3", " 1.0", "1.0 ", "0.١", "--1.0"]
)
def test_decimal_grammar_is_the_source_ascii_grammar(text: str) -> None:
    with pytest.raises(ValueError, match="Decimal phase"):
        Angle.from_rounded_decimal_half_turns(text)


def test_power_grid_and_the_source_subnormal_double_boundary() -> None:
    assert Angle.from_power_of_two_half_turns(0) == Angle(0, 1 << 63)
    assert Angle.from_power_of_two_half_turns(-127) == Angle(1, 0)
    assert Angle.from_power_of_two_half_turns(-128) == Angle()
    assert Angle.from_power_of_two_half_turns(1) == Angle()
    assert Angle(1, 0).clifford_hierarchy_level() == 128
    assert Angle(0, 1 << 63).clifford_hierarchy_level() == 1
    with pytest.raises(ValueError, match="below"):
        Angle.from_half_turns_exact_double(ldexp(1.0, -128))
    # The locked routine halves the double before checking its low bits.
    # The smallest subnormal therefore underflows to zero instead of rejecting.
    assert Angle.from_half_turns_exact_double(5e-324) == Angle()
    assert Angle.from_half_turns_exact_double(-5e-324) == Angle()


def test_raw_words_and_multiplier_validation() -> None:
    for low, high in ((-1, 0), (0, 1 << 64), (True, 0)):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            Angle(low, high)
    for factor in (-1, 1 << 64, True):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            Angle(1, 0) * factor
    assert Angle(1, 0) + Angle((1 << 64) - 1, 0) == Angle(0, 1)
    assert -Angle(1, 0) == Angle((1 << 64) - 1, (1 << 64) - 1)
    assert Angle.from_integer_half_turns(-3) == Angle(0, 1 << 63)
    assert Angle(1, 0).plus_integer_half_turns(2) == Angle(1, 0)


@pytest.mark.parametrize("negative", [False, True])
def test_numeric_boundary_preserves_the_small_wrapped_phase(negative: bool) -> None:
    angle = -Angle(1, 0) if negative else Angle(1, 0)
    theta = pi * angle.signed_half_turns_float()
    gate = Instruction("phase", (0,), {"theta": theta})
    circuit = (
        DynamicCircuit(1, dtype=torch.complex128)
        .x(0)
        .append(FieldOperation(1, (gate,)))
    )
    state = run_dynamic(
        circuit, shots=1, seed=41, strategy="batched"
    ).final_states.reshape(-1)
    assert state[0] == 0
    assert state[1].real == 1
    assert state[1].imag.item() == pytest.approx(sin(theta), rel=1e-15, abs=1e-53)
    assert state[1].imag.item() != 0
