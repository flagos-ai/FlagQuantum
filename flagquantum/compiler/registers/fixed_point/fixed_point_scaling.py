# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 ScaleIntByReal supplies the register and cost contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2007.07391#page=70

"""Term-truncated products of unsigned fixed-point fractions."""

from ....core.ir import Instruction
from ..arithmetic.controlled_addition import (
    controlled_add_adjoint_ir,
    controlled_add_ir,
)
from ..bits.bitwise import _validate_wires
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir


def scale_integer_workspace_size(real_width: int, integer_width: int) -> int:
    """Return the size of each reusable loaded-word and carry pool."""
    if (
        type(real_width) is not int
        or type(integer_width) is not int
        or not 1 <= integer_width <= real_width
    ):
        raise ValueError(
            "Fixed-point scaling requires 1 <= integer width <= real width"
        )
    return real_width - 1 if integer_width > 1 else 0


def scale_integer_by_real_ir(
    real: tuple[int, ...],
    integer: tuple[int, ...],
    result: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    copy_bit: int = 0,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compute the paper's truncated product into a zero result, or erase it.

    Registers are big endian. The real is a pure fraction; the output has
    len(real) - len(integer) fractional bits. The caller owns two equally
    sized clean pools with disjoint classical slots. The initial controlled
    copy is erased with a reusable slot after the reversed additions.
    """
    size = scale_integer_workspace_size(len(real), len(integer))
    return _emit_truncated_scale(
        real,
        integer,
        result,
        loaded,
        carries,
        classical_bits,
        first_width=len(real) - len(integer),
        omitted_low_bits=0,
        size=size,
        copy_bit=copy_bit,
        adjoint=adjoint,
    )


def fixed_point_product_workspace_size(width: int) -> int:
    """Return each reusable work-pool width for equal-width fractions."""
    if type(width) is not int or width < 1:
        raise ValueError("Fixed-point product requires a positive input width")
    return width - 1 if width > 2 else 0


def fixed_point_product_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    result: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    copy_bit: int = 0,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compute or erase the term-truncated product of two pure fractions.

    All three words are big endian with equal width and all bits fractional.
    The forward result starts at zero. Erase expects the computed value.
    The lowest multiplier bit contributes a zero term. Work finishes clean.
    """
    size = fixed_point_product_workspace_size(len(left))
    if len(right) != len(left):
        raise ValueError("Fixed-point product requires equal-width inputs")
    return _emit_truncated_scale(
        left,
        right,
        result,
        loaded,
        carries,
        classical_bits,
        first_width=1,
        omitted_low_bits=1,
        size=size,
        copy_bit=copy_bit,
        adjoint=adjoint,
    )


def _emit_truncated_scale(
    real: tuple[int, ...],
    multiplier: tuple[int, ...],
    result: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    first_width: int,
    omitted_low_bits: int,
    size: int,
    copy_bit: int,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    if len(result) != len(real) or len(loaded) != size or len(carries) != size:
        raise ValueError(
            "Fixed-point scaling requires the pinned result and work widths"
        )
    if (
        len(classical_bits) != 2 * size
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(bit) is not int or bit < 0 for bit in (*classical_bits, copy_bit))
    ):
        raise ValueError("Fixed-point scaling requires valid distinct pool slots")
    if type(adjoint) is not bool:
        raise ValueError("Fixed-point scaling adjoint must be boolean")
    _validate_wires((*real, *multiplier, *result, *loaded, *carries))

    active = tuple(reversed(multiplier))[omitted_low_bits:]
    if not active:
        return ()
    pairs = tuple(
        zip(real[:first_width], result[len(result) - first_width :], strict=True)
    )
    control = active[0]
    if adjoint:
        copy = tuple(
            gate
            for wire, output in reversed(pairs)
            for gate in and_uncompute_ir(control, wire, output, copy_bit)
        )
    else:
        copy = tuple(
            gate
            for wire, output in pairs
            for gate in and_compute_ir(control, wire, output)
        )

    emit = controlled_add_adjoint_ir if adjoint else controlled_add_ir
    steps = tuple(enumerate(active[1:], start=first_width + 1))
    additions: list[Instruction] = []
    for width, control in reversed(steps) if adjoint else steps:
        additions.extend(
            emit(
                control,
                real[:width],
                result[-width - 1 :],
                loaded[:width],
                carries[:width],
                classical_bits[:width],
                classical_bits[size : size + width],
            )
        )
    return (*additions, *copy) if adjoint else (*copy, *additions)
