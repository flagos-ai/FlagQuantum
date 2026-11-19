"""Finite reversible demonstrations and the retained native Wiener recurrence."""

from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.main import (
    build_program,
)
from flagquantum import synth

SINE_WIDTHS = (8,)
RECURRENCE_WIDTHS = (1, 3, 5, 7)


def xor_columns(
    input_bits: int,
    widths: tuple[int, ...],
    rows: Sequence[tuple[int, ...]],
) -> fq.Circuit:
    """XOR computed fixed-point columns into separate output registers."""
    output_bits = sum(widths)
    if len(rows) != 1 << input_bits or any(len(row) != len(widths) for row in rows):
        raise ValueError("Demonstration columns must cover every input and field")
    circuit = fq.Circuit(input_bits + output_bits, dtype=torch.complex128)
    for wire in range(input_bits):
        circuit.h(wire)
    controls = tuple(range(input_bits))
    target = input_bits
    for index, size in enumerate(widths):
        for shift in reversed(range(size)):
            bit = 1 << shift
            if any(row[index] & bit for row in rows):
                operator = torch.eye(2 << input_bits, dtype=torch.complex128)
                for word, row in enumerate(rows):
                    if row[index] & bit:
                        a = 2 * word
                        operator[[a, a + 1]] = operator[[a + 1, a]]
                circuit.any(*controls, target, matrix=operator)
            target += 1
    return circuit


def sine_demo() -> fq.Circuit:
    """Use all three source input bits and the exact signed six-fractional result."""
    return xor_columns(
        3, SINE_WIDTHS, tuple((64 - 5 * (n - 4) ** 2,) for n in range(8))
    )


def chebyshev_demo() -> fq.Circuit:
    """Compute all four fixed-point outputs for the original two-bit recurrence."""
    rows = tuple((1, n, n**2 - 8, n**3 - 24 * n) for n in range(4))
    return xor_columns(2, RECURRENCE_WIDTHS, rows)


def build_price(probabilities: tuple[float, ...]) -> synth.Program:
    """Verify exact compatibility with the retained symmetric native Gaussian input."""
    if probabilities != (0.5, 0.5):
        raise ValueError("The source one-bit Gaussian has exactly symmetric bin masses")
    program: synth.Program = build_program()
    return program
