# SPDX-License-Identifier: Apache-2.0
"""Evaluate an alternating polynomial from coherent coefficient registers."""

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires, xor_ir
from .fixed_point_horner import (
    register_horner_step_ir,
    register_horner_step_requirements,
)


def register_polynomial_requirements(argument: int, width: int) -> tuple[int, int]:
    """Count three clean Horner words and reusable multiply/add scratch."""
    work, slots = register_horner_step_requirements(argument, width)
    return 3 * width + work, slots


def register_polynomial_ir(
    argument: tuple[int, ...],
    coefficients: tuple[tuple[int, ...], ...],
    output: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    argument_fraction_bits: int,
) -> tuple[Instruction, ...]:
    """XOR c0-x*(c1-x*(c2-x*c3)) with truncation at each Horner step.

    Coefficients are unsigned big-endian words with a common fixed-point
    scale. Each intermediate wraps at that width. Argument fractional bits
    determine product truncation. The output receives the low result bits,
    zero-extended if wider. All inputs and caller work are preserved.
    Reapply this operation to clear a matching output.
    """
    if len(coefficients) != 4 or not coefficients[0] or not output:
        raise ValueError(
            "Register polynomial requires four nonempty coefficients and output"
        )
    width = len(coefficients[0])
    if any(len(word) != width for word in coefficients):
        raise ValueError("Register polynomial coefficient widths must match")
    work, slots = register_polynomial_requirements(len(argument), width)
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError("Register polynomial requires matching work and slots")
    _validate_wires(
        (
            *argument,
            *(wire for word in coefficients for wire in word),
            *output,
            *workspace,
        )
    )
    history = tuple(workspace[i * width : (i + 1) * width] for i in range(3))
    scratch = workspace[3 * width :]
    steps = tuple(
        zip(
            (coefficients[-1], *history[:-1]),
            reversed(coefficients[:-1]),
            history,
            strict=True,
        )
    )

    def evaluate(adjoint: bool) -> tuple[Instruction, ...]:
        return tuple(
            gate
            for previous, coefficient, result in (reversed(steps) if adjoint else steps)
            for gate in register_horner_step_ir(
                argument,
                previous,
                coefficient,
                result,
                scratch,
                classical_bits,
                argument_fraction_bits=argument_fraction_bits,
                adjoint=adjoint,
            )
        )

    retained = min(width, len(output))
    return (
        *evaluate(False),
        *xor_ir(history[-1][-retained:], output[-retained:]),
        *evaluate(True),
    )
