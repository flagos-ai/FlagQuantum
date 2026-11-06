# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 InvertRealNumber supplies the reciprocal resource model.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Numerical construction: https://arxiv.org/pdf/1511.08253#page=30

"""Fixed-point reciprocal via a power-of-two seed and widened Newton steps."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, xor_ir
from .reciprocal_newton import reciprocal_newton_requirements, reciprocal_newton_step_ir
from .reciprocal_seed import reciprocal_seed_ir


def reciprocal_requirements(
    input_width: int, precision: int, *, source_fraction_bits: int
) -> tuple[int, int]:
    """Return the history plus shared scratch width and the classical slot count."""
    if (
        type(input_width) is not int
        or type(source_fraction_bits) is not int
        or not 0 <= source_fraction_bits < input_width
    ):
        raise ValueError("Reciprocal requires a nonempty source integer part")
    integer_width = input_width - source_fraction_bits
    if type(precision) is not int or precision < integer_width:
        raise ValueError("Reciprocal precision must cover every source integer bit")
    steps = (precision - 1).bit_length()
    scratch, slots = (
        reciprocal_newton_requirements(input_width, precision)
        if steps
        else (integer_width, integer_width - 1)
    )
    return (steps + 1) * precision + scratch, slots


def reciprocal_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    precision: int,
) -> tuple[Instruction, ...]:
    """XOR a reciprocal approximation into a big-endian pure fraction.

    For source a >= 1, use ceil(log2(precision)) widened Newton steps
    with truncation to precision bits after each step. The result may
    retain any nonempty prefix of that estimate. This is a lower
    approximation, including at a=1. Below one the output mask is zero.
    History and scratch must start clean and are restored before return.
    """
    required, slots = reciprocal_requirements(
        len(source), precision, source_fraction_bits=source_fraction_bits
    )
    if not 1 <= len(result) <= precision:
        raise ValueError(
            "Reciprocal requires a nonempty result within working precision"
        )
    if len(workspace) != required or len(classical_bits) != slots:
        raise ValueError(f"Reciprocal requires {required} work wires and {slots} slots")
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Reciprocal requires distinct nonnegative classical slots")
    _validate_wires((*source, *result, *workspace))

    steps = (precision - 1).bit_length()
    end = (steps + 1) * precision
    history = tuple(workspace[i : i + precision] for i in range(0, end, precision))
    scratch = workspace[end:]
    integer_width = len(source) - source_fraction_bits
    seed_args = (
        source,
        history[0],
        scratch[0],
        scratch[1:integer_width],
        classical_bits[: integer_width - 1],
    )
    seed = reciprocal_seed_ir(*seed_args, source_fraction_bits=source_fraction_bits)
    updates = tuple(
        reciprocal_newton_step_ir(
            source,
            previous,
            following,
            scratch,
            classical_bits,
            source_fraction_bits=source_fraction_bits,
        )
        for previous, following in zip(history[:-1], history[1:], strict=True)
    )
    copy = xor_ir(history[-1][: len(result)], result)
    erase_seed = reciprocal_seed_ir(
        *seed_args, source_fraction_bits=source_fraction_bits, adjoint=True
    )
    return (
        *seed,
        *(gate for update in updates for gate in update),
        *copy,
        *(gate for update in reversed(updates) for gate in update),
        *erase_seed,
    )
