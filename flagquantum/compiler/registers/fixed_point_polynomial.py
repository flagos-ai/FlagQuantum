# SPDX-License-Identifier: Apache-2.0
# Horner history and fixed-point arithmetic from Appendix B:
# https://arxiv.org/pdf/1805.12445#page=7

"""Reversible Horner evaluation with explicit clean history."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, xor_constant_ir, xor_ir
from .fixed_point_horner import (
    fixed_point_horner_step_ir,
    fixed_point_horner_step_requirements,
)


def fixed_point_polynomial_requirements(width: int, degree: int) -> tuple[int, int]:
    """Return clean history plus shared arithmetic scratch and slot count."""
    step_work, slots = fixed_point_horner_step_requirements(width)
    if type(degree) is not int or degree < 0:
        raise ValueError("Polynomial requires a nonnegative degree")
    if degree == 0:
        return 0, 0
    return (degree + 1) * width + step_work, slots


def fixed_point_polynomial_ir(
    argument: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    argument_fraction_bits: int,
    coefficients: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR the Horner result into an arbitrary big-endian result word.

    The argument is unsigned. Coefficients are signed words in descending
    degree order, in the result's units. Each intermediate keeps the same
    two's-complement format and uses the pre-truncated Horner step. Clean
    history and arithmetic scratch are restored, so reapplying erases the
    result of a previous application.
    """
    width = len(result)
    f = argument_fraction_bits
    if not argument or type(f) is not int or not 0 <= f <= len(argument):
        raise ValueError("Polynomial requires a valid argument format")
    work, slots = fixed_point_polynomial_requirements(width, len(coefficients) - 1)
    limit = 1 << (width - 1)
    if any(type(c) is not int or not -limit <= c < limit for c in coefficients):
        raise ValueError("Polynomial coefficients must fit the signed result width")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(f"Polynomial requires {work} work wires and {slots} slots")
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Polynomial requires distinct nonnegative slots")
    _validate_wires((*argument, *result, *workspace))

    if len(coefficients) == 1:
        return xor_constant_ir(result, coefficients[0] % (1 << width))
    history_width = len(coefficients) * width
    history = tuple(workspace[i : i + width] for i in range(0, history_width, width))
    scratch = workspace[history_width:]
    load = xor_constant_ir(history[0], coefficients[0] % (1 << width))
    forward: list[tuple[Instruction, ...]] = []
    backward: list[tuple[Instruction, ...]] = []
    for previous, following, coefficient in zip(
        history[:-1], history[1:], coefficients[1:], strict=True
    ):
        args = (argument, previous, following, scratch, classical_bits)
        forward.append(
            fixed_point_horner_step_ir(
                *args, argument_fraction_bits=f, coefficient=coefficient
            )
        )
        backward.append(
            fixed_point_horner_step_ir(
                *args, argument_fraction_bits=f, coefficient=coefficient, adjoint=True
            )
        )
    return (
        *load,
        *(gate for step in forward for gate in step),
        *xor_ir(history[-1], result),
        *(gate for step in reversed(backward) for gate in step),
        *load,
    )
