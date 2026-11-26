"""Record the unchanged exact-angle helper through both original consumers."""

import argparse
import csv
import hashlib
import importlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


def cases() -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = [
        ("bool_false", False),
        ("bool_true", True),
        ("integer_zero", 0),
        ("integer_negative", -3),
        ("integer_huge", 1 << 500),
        ("float_tenth", 0.1),
        ("string_tenth", "0.1"),
        ("decimal_tenth", Decimal("0.1")),
        ("fraction_tenth", Fraction(1, 10)),
        ("fraction_string", " 1 / 3 "),
        ("underscored_string", "1_000/3"),
        ("integer_string", "1000"),
        ("negative_decimal", Decimal("-20.125")),
        ("negative_float", -2.125),
        ("hex_float_value", float.fromhex("0x1.0000000000001p-1")),
        ("smallest_float", float.fromhex("0x0.0000000000001p-1022")),
        ("invalid_string", "not_a_fraction"),
        ("zero_denominator", "1/0"),
        ("nan", float("nan")),
        ("infinity", float("inf")),
        ("decimal_nan", Decimal("NaN")),
        ("decimal_infinity", Decimal("Infinity")),
        ("complex", complex(1, 0)),
        ("none", None),
        ("object", object()),
    ]
    values += [
        (f"level_{level}", Fraction(1, 1 << (level - 1))) for level in range(1, 129)
    ]
    values += [(f"tie_{n}", Fraction(n, 1 << 128)) for n in range(-9, 10)]
    values += [(f"wrap_{n}", 2 + Fraction(n, 1 << 128)) for n in range(-5, 6)]
    values += [(f"large_fraction_{n}", Fraction((1 << 600) + n, 7)) for n in range(7)]
    return values


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    expected = {
        row["path"]: row["git_blob"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    names = (
        "kickmix/src/kickmix/py/val/angle.pybind.cc",
        "kickmix/src/kickmix/py/val/angle.pybind.h",
        "kickmix/src/kickmix/util/fixed_precision_angle_128.h",
        "kickmix/src/kickmix/py/circuit/circuit_builder.pybind.cc",
        "kickmix/src/kickmix/py/interactive_simulator.pybind.cc",
        "kickmix/src/kickmix/py/circuit/circuit_builder_test.py",
        "kickmix/src/kickmix/py/interactive_simulator_test.py",
    )
    verified = {}
    for name in names:
        raw = (source / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[name]:
            raise ValueError(f"Frozen angle conversion source differs for {name}")
        verified[name] = blob
    km = importlib.import_module("kickmix")
    if km.__version__ != "0.1" or not hasattr(km, "Simulator"):
        raise RuntimeError("The pinned native extension and Simulator are required")
    sim = km.Simulator(batch_size=1, debug_print=False, seed=1)
    rows = []
    for name, value in cases():
        sim.write_shot_phase(0, Fraction(1, 4))
        builder = km.CircuitBuilder()
        try:
            sim.write_shot_phase(0, value)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            if sim.read_shot_phase(0) != Fraction(1, 4):
                raise AssertionError(
                    "Invalid source conversion changed the stored phase"
                ) from error
            try:
                builder.z_pow(km.q(0), value)
            except (
                TypeError,
                ValueError,
                OverflowError,
                ZeroDivisionError,
            ) as builder_error:
                if type(builder_error) is not type(error) or str(builder_error) != str(
                    error
                ):
                    raise AssertionError(
                        "Source converter consumers disagree"
                    ) from builder_error
            else:
                raise AssertionError(
                    "Invalid source builder conversion unexpectedly passed"
                ) from error
            rows.append(
                [
                    name,
                    None,
                    type(error).__name__,
                    str(error),
                    str(builder.finish_circuit()),
                ]
            )
            continue
        phase = sim.read_shot_phase(0)
        builder.z_pow(km.q(0), value)
        units = phase * (1 << 127)
        if units.denominator != 1:
            raise AssertionError("Source phase is outside its exact grid")
        rows.append(
            [name, str(units.numerator), None, None, str(builder.finish_circuit())]
        )
    output.write_text(
        json.dumps(
            {
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": verified,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source_root, args.inventory, args.output)


if __name__ == "__main__":
    main()
