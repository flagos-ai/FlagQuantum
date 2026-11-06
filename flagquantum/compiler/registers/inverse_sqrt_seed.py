# SPDX-License-Identifier: Apache-2.0
# First-set-bit initial estimate from Appendix C Eqs. C1 and C2:
# https://arxiv.org/pdf/1805.12445#page=8

"""Power-of-two-selected initial estimate for inverse square root."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, controlled_xor_constant_ir, xor_ir
from .controlled_addition import controlled_add_adjoint_ir, controlled_add_ir


def _seed_format(n: int, f: int, b: int) -> tuple[int, int]:
    if type(n) is not int or n < 1 or type(f) is not int or not 0 <= f <= n:
        raise ValueError("Inverse square root seed requires a valid source format")
    if type(b) is not int or b < 0:
        raise ValueError(
            "Inverse square root seed requires nonnegative result precision"
        )
    smallest_shift = (f - n + 1) // 2
    guard = max(0, f + 1 - 3 * smallest_shift - b)
    return b + guard + f // 2 + 1, guard


def inverse_sqrt_seed_requirements(
    source_width: int, *, source_fraction_bits: int, result_fraction_bits: int
) -> tuple[int, int]:
    """Return work and slot counts for the exact shifted seed construction."""
    n = source_width
    width, _ = _seed_format(n, source_fraction_bits, result_fraction_bits)
    slots = n + width - 1
    return n + 1 + width + slots, slots


def inverse_sqrt_seed_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    result_fraction_bits: int,
    tuned: bool = False,
) -> tuple[Instruction, ...]:
    """XOR the truncated C1 or C2 estimate into an unsigned big-endian result.

    A zero source yields a zero mask. The result has b fractional bits
    and at least floor(f/2)+1 integer bits, where f is the source fraction
    width. Both source and initially clean work are restored. The optional
    C2 seed may exceed the true inverse square root.
    """
    if type(tuned) is not bool:
        raise ValueError("Inverse square root tuning must be boolean")
    n, f, b = len(source), source_fraction_bits, result_fraction_bits
    width, guard = _seed_format(n, f, b)
    if len(result) < b + f // 2 + 1:
        raise ValueError("Inverse square root seed needs more result integer bits")
    work, slots = inverse_sqrt_seed_requirements(
        n, source_fraction_bits=f, result_fraction_bits=b
    )
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Inverse square root seed requires {work} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Inverse square root seed requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    found = workspace[0]
    selectors = workspace[1 : n + 1]
    value = workspace[n + 1 : n + 1 + width]
    loaded = workspace[n + 1 + width : 2 * n + 1 + width]
    carries = workspace[2 * n + 1 + width :]
    flip = Instruction("x", (found,))
    select = tuple(
        gate
        for bit, selected in zip(source, selectors, strict=True)
        for gate in (
            flip,
            Instruction("ccx", (bit, found, selected)),
            flip,
            Instruction("cx", (selected, found)),
        )
    )

    compute: list[Instruction] = []
    undo: list[tuple[Instruction, ...]] = []
    for i, selected in enumerate(selectors):
        k = (f - n + 1 + i) // 2
        if tuned and k:
            numerator = 1613 if k < 0 else 1620
            constant_word = (numerator << (b + guard + k)) // 1000
        else:
            constant_word = 3 << (b + guard + k - 1)
        constant = controlled_xor_constant_ir(selected, value, constant_word)
        shift = b + guard + 3 * k - 1 - f
        shifted_target = value[: width - shift]
        carry_count = len(shifted_target) - 1
        args = (
            selected,
            source[i:],
            shifted_target,
            loaded[: n - i],
            carries[:carry_count],
            classical_bits[: n - i],
            classical_bits[n : n + carry_count],
        )
        compute.extend((*constant, *controlled_add_adjoint_ir(*args)))
        undo.append((*controlled_add_ir(*args), *constant))

    significant = width - guard
    copied = min(len(result), significant)
    copy = xor_ir(value[significant - copied : significant], result[-copied:])
    return (
        *select,
        *compute,
        *copy,
        *(gate for block in reversed(undo) for gate in block),
        *reversed(select),
    )
