"""Generate source observations without importing a native implementation."""

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from modular_integer_reference import ModularCase

SOURCE_BLOBS = {
    "mod_int.cc": "6d6af13a5cd0fe16c9ca54aa51c3d45cf40ec9e2",
    "mod_int.h": "7d2a8926061335b16c5fbc8cfc8c56266aa2d8bc",
    "mod_int.test.cc": "9ecb7e800c1ff76476ea833b47cc3bb495fdeba2",
    "fixed_width_int.cc": "50a2300a311befb075bd6680287fa3e0e36bacc2",
    "fixed_width_int.h": "7057645bae714c72deb5b60dca85589bd4a9c8a7",
    "word_ops.h": "b0cab29ebcd7e616b6a40e7e4d99eb72bf0c9a45",
    "bit.h": "ea1c426299ab2ed0a7da662f629b133a106228e2",
}


def cases() -> Iterator[ModularCase]:
    rng = random.Random(1084)
    operations = (
        "inspect zero signed text word word_move copy move assign assign_move equal "
        "add add_scalar sub mul mul_self mul_scalar square divide divide_move "
        "divide_scalar divide_self divide_self_move negate increment inverse double halve "
        "left64 right64 left32 right32 left_u64 right_u64 left_u32 right_u32 "
        "random randomize"
    ).split()
    for width in (1, 2, 3, 4, 7, 63, 64, 65, 127, 128, 129, 256):
        for index in range(12):
            modulus = rng.randrange(1 << (width - 1), 1 << width)
            value, other = rng.randrange(modulus), rng.randrange(modulus)
            scalar = rng.randrange(-31, 32)
            for operation in operations:
                shift = scalar if "_u" not in operation else abs(scalar)
                yield ModularCase(
                    operation,
                    width,
                    hex(value),
                    width,
                    hex(other),
                    width,
                    hex(modulus),
                    hex(max(1, modulus - 2)),
                    shift,
                    index,
                    str(value * modulus + other),
                )
    # Exhaust reduced and unreduced small storage, including even/nonunit division.
    for modulus in range(1, 16):
        width = modulus.bit_length()
        for value in range(1 << width):
            for other in range(modulus):
                for operation in ("add", "sub", "mul", "divide"):
                    yield ModularCase(
                        operation,
                        width,
                        hex(value),
                        width,
                        hex(other),
                        width,
                        hex(modulus),
                        hex(modulus),
                        0,
                        0,
                        "0",
                    )
    for width in (3, 64, 65, 128):
        modulus = (1 << width) - 1
        for scalar in (-(1 << 63), -10, -1, 0, 1, 7, 8, 10, (1 << 63) - 1):
            for operation in ("signed", "add_scalar", "divide_scalar", "mul_scalar"):
                if operation == "mul_scalar" and scalar == -(1 << 63):
                    continue  # Original signed overflow has a separate sanitizer probe.
                yield ModularCase(
                    operation,
                    width,
                    "0x1",
                    width,
                    "0x1",
                    width,
                    hex(modulus),
                    hex(modulus),
                    scalar,
                    0,
                    "0",
                )
        for text in ("-1", "-10", "bad", "0", "7", "8", "0xff", "0b101"):
            yield ModularCase(
                "text",
                width,
                "0x0",
                width,
                "0x0",
                width,
                hex(modulus),
                hex(modulus),
                0,
                0,
                text,
            )
        for input_width in (1, 3, 63, 64, 65, 129, 256):
            for operation in ("word", "word_move"):
                yield ModularCase(
                    operation,
                    input_width,
                    hex((1 << input_width) - 1),
                    width,
                    "0x0",
                    width,
                    hex(modulus),
                    hex(modulus),
                    0,
                    0,
                    "0",
                )
    # The original unit-test constants exercise its wide constructor and division.
    modulus = 503770467140461526195448602895
    for operation in operations:
        yield ModularCase(
            operation,
            modulus.bit_length(),
            hex(267758050305935043618143975821),
            modulus.bit_length(),
            hex(393836260909983489752333563576),
            modulus.bit_length(),
            hex(modulus),
            hex(modulus),
            2,
            1084,
            "134888598083240094192267100302247074140311888793175110601800",
        )
    for width, raw, modulus_width, modulus in (
        (0, 0, 3, 7),
        (3, 1, 4, 7),
        (4, 1, 3, 7),
        (3, 8, 3, 7),
        (3, 63, 65, (1 << 64) + 13),
        (65, (1 << 67) + 9, 3, 7),
        (3, 1, 0, 0),
        (0, 0, 0, 0),
    ):
        for operation in ("inspect", "word", "word_move"):
            yield ModularCase(
                operation,
                width,
                hex(raw),
                3,
                "0x1",
                modulus_width,
                hex(modulus),
                "0x7",
                0,
                0,
                "0",
            )


def regenerate(source_root: Path, output: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / "kickmix/util" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source blob mismatch for {name}")
    inputs = list(cases())
    with tempfile.TemporaryDirectory(prefix="fq-modular-integer-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source_root),
                str(Path(__file__).with_name("modular_integer_oracle.cc")),
                str(source_root / "kickmix/util/mod_int.cc"),
                str(source_root / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(case.source_input() for case in inputs),
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(inputs):
        raise ValueError("Original modular oracle returned an incomplete stream")
    with output.open("w") as stream:
        for case, line in zip(inputs, lines, strict=True):
            stream.write(json.dumps([case, line], separators=(",", ":")) + "\n")
    return len(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Recorded {regenerate(args.source_root, args.output)} original observations")


if __name__ == "__main__":
    main()
