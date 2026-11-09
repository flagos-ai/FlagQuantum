# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QFTPhaseGradient.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/qft_phase_gradient.py

"""Recursive Fourier transform with an explicit phase-gradient register."""

from ...compiler.registers.arithmetic.integer_product import (
    integer_product_requirements,
)
from ...compiler.registers.arithmetic.product_accumulation import (
    product_accumulation_ir,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...core.ir import Instruction


def qft_phase_gradient_requirements(width: int) -> tuple[int, int]:
    """Return the clean work and classical slots reused across recursion."""
    if type(width) is not int or width < 1:
        raise ValueError("Phase-gradient QFT requires positive width")
    if width == 1:
        return 0, 0
    left = width // 2
    _, product = integer_product_requirements(left, width - left)
    return product + width - 1, width - 1


def qft_phase_gradient_ir(
    data: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    with_reverse: bool = True,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Bind the source recursion on equal-width big-endian data and gradient.

    The negative phase-gradient eigenstate is preserved and induces QFT on
    data. For other gradient inputs this applies the full joint operator.
    Work starts and finishes clean. Classical slots can be reused.
    """
    work, slots = qft_phase_gradient_requirements(len(data))
    if len(gradient) != len(data) or len(workspace) != work:
        raise ValueError(
            "Phase-gradient QFT requires matching registers and clean work"
        )
    if (
        len(classical_bits) != slots
        or len(set(classical_bits)) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Phase-gradient QFT requires distinct nonnegative slots")
    if type(with_reverse) is not bool or type(adjoint) is not bool:
        raise ValueError("Phase-gradient QFT direction and reversal must be boolean")
    _validate_wires((*data, *gradient, *workspace))

    def emit(
        word: tuple[int, ...], phase_word: tuple[int, ...]
    ) -> tuple[Instruction, ...]:
        width = len(word)
        if width == 1:
            return (Instruction("h", word),)
        middle = width // 2
        left, right = word[:middle], word[middle:]
        a = emit(left, phase_word[:middle])
        b = emit(right, phase_word[: len(right)])
        _, product = integer_product_requirements(middle, len(right))
        add = product_accumulation_ir(
            left[::-1],
            right,
            phase_word,
            workspace[:product],
            workspace[product : product + width - 1],
            classical_bits[: width - 1],
            adjoint=adjoint,
        )
        return (*b, *add, *a) if adjoint else (*a, *add, *b)

    swaps = (
        tuple(
            Instruction("swap", (data[i], data[-1 - i])) for i in range(len(data) // 2)
        )
        if with_reverse
        else ()
    )
    circuit = emit(data, gradient)
    return (*reversed(swaps), *circuit) if adjoint else (*circuit, *swaps)
