# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 ScaleIntByReal supplies the register and cost contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2007.07391#page=70

"""Term-truncated scaling of an unsigned integer by a fixed-point fraction."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires
from .controlled_addition import controlled_add_adjoint_ir, controlled_add_ir


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
    _validate_wires((*real, *integer, *result, *loaded, *carries))

    fractional = len(real) - len(integer)
    pairs = tuple(zip(real[:fractional], result[len(integer) :], strict=True))
    control = integer[-1]
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
    steps = tuple(enumerate(reversed(integer[:-1]), start=1))
    additions: list[Instruction] = []
    for shift, control in reversed(steps) if adjoint else steps:
        width = fractional + shift
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
