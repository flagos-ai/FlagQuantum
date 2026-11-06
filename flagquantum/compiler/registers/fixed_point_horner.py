# SPDX-License-Identifier: Apache-2.0
# Fixed-point multiply-accumulate from Appendices A and B:
# https://arxiv.org/pdf/1805.12445#page=7

"""Horner multiply-add using shifted, pre-truncated contributions."""

from ...core.ir import Instruction
from .arithmetic import add_constant_adjoint_ir, add_constant_ir
from .bitwise import _validate_wires
from .controlled_addition import controlled_add_adjoint_ir, controlled_add_ir


def fixed_point_horner_step_requirements(width: int) -> tuple[int, int]:
    """Return one reusable load word, a carry bank, and their slot count."""
    if type(width) is not int or width < 1:
        raise ValueError("Horner multiply-add requires positive result width")
    work = 2 * width - 1
    return work, work


def fixed_point_horner_step_ir(
    argument: tuple[int, ...],
    previous: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    argument_fraction_bits: int,
    coefficient: int,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Accumulate a pre-truncated product plus a signed coefficient.

    The unsigned argument has explicit fractional width. Big-endian
    previous and result words share a two's-complement format, and the
    coefficient is a signed integer in result ulps. An occupied result is
    updated modulo its width. Reverse the arithmetic for the adjoint.
    """
    width = len(result)
    work, slots = fixed_point_horner_step_requirements(width)
    f = argument_fraction_bits
    if not argument or type(f) is not int or not 0 <= f <= len(argument):
        raise ValueError("Horner multiply-add requires a valid argument format")
    if len(previous) != width:
        raise ValueError(
            "Horner multiply-add requires matching previous and result widths"
        )
    limit = 1 << (width - 1)
    if type(coefficient) is not int or not -limit <= coefficient < limit:
        raise ValueError("Horner coefficient must fit the signed result width")
    if type(adjoint) is not bool:
        raise ValueError("Horner adjoint must be boolean")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Horner multiply-add requires {work} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Horner multiply-add requires distinct nonnegative slots")
    _validate_wires((*argument, *previous, *result, *workspace))

    load, carries = workspace[:width], workspace[width:]
    load_bits, carry_bits = classical_bits[:width], classical_bits[width:]
    add_constant = add_constant_adjoint_ir if adjoint else add_constant_ir
    offset = (
        add_constant(result, coefficient, load, carries, carry_bits)
        if coefficient
        else ()
    )

    blocks: list[tuple[Instruction, ...]] = []
    positions = range(width - 1, -1, -1) if adjoint else range(width)
    for j in positions:
        shift = j - f
        destination = result[: width - max(0, shift)]
        significant = argument[: max(0, len(argument) + min(0, shift))]
        operand = significant[-len(destination) :]
        if not operand:
            continue
        subtract = (j == width - 1) ^ adjoint
        emit = controlled_add_adjoint_ir if subtract else controlled_add_ir
        blocks.append(
            emit(
                previous[-1 - j],
                operand,
                destination,
                load[: len(operand)],
                carries[: len(destination) - 1],
                load_bits[: len(operand)],
                carry_bits[: len(destination) - 1],
            )
        )
    product = tuple(gate for block in blocks for gate in block)
    return (*offset, *product) if adjoint else (*product, *offset)
