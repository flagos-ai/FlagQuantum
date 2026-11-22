# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# Adapted from frozen Kickmix util/circuit_testing.test.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Shared source-exact diagram expectations using the existing native renderer."""

import json
from itertools import zip_longest
from pathlib import Path

from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from flagquantum.drawer.reversible_phase_diagram import render_circuit_text


def undent(text: str) -> str:
    text = text.lstrip("\n").rstrip("\n ")
    sentinel = 1 << 64
    minimum = sentinel
    current = 0
    for character in text:
        if character == "\n":
            current = 0
        elif character == " " and current != sentinel:
            current += 1
        else:
            if character != "\r":
                minimum = min(minimum, current)
            current = sentinel
    output = []
    skip = minimum
    for character in text:
        if character == "\n":
            skip = minimum
            output.append(character)
        elif character == " " and skip:
            skip -= 1
        else:
            skip = 0
            output.append(character)
    return "".join(output)


def split_lines(text: str) -> list[str]:
    return text.split("\n")


def diagram_failure(actual: str, expected: str) -> bytes | None:
    expected = undent(expected)
    if actual == expected:
        return None
    actual_lines = [line.encode() for line in split_lines(actual)]
    expected_lines = [line.encode() for line in split_lines(expected)]
    height = max(len(actual_lines), len(expected_lines))
    actual_lines.extend([b""] * (height - len(actual_lines)))
    expected_lines.extend([b""] * (height - len(expected_lines)))
    differences = []
    for left, right in zip_longest(actual_lines, expected_lines, fillvalue=b""):
        line = bytearray(b"        ")
        for actual_byte, expected_byte in zip_longest(left, right, fillvalue=-1):
            if actual_byte != expected_byte:
                line.extend("█".encode())
            else:
                line.append(actual_byte)
        differences.append(bytes(line) + b"\n")
    return (
        b"Actual diagram:\n"
        + b"".join(b"        " + line + b"\n" for line in actual_lines)
        + b"\nDiff:\n"
        + b"".join(differences)
    )


def expect_circuit_has_text_diagram(circuit: KmxCircuit, expected: str) -> None:
    failure = diagram_failure(render_circuit_text(circuit), expected)
    if failure is not None:
        raise AssertionError(failure)


def circuit_with_every_operation() -> KmxCircuit:
    path = Path(__file__).with_name("data") / "circuit_testing.json"
    data: object = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("circuit"), list)
        or not data["circuit"]
    ):
        raise ValueError("Invalid complete circuit fixture")
    kmx = data["circuit"][0]
    if not isinstance(kmx, str):
        raise ValueError("Invalid complete circuit text")
    return read_kmx_circuit(kmx)
