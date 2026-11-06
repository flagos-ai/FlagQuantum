# SPDX-License-Identifier: Apache-2.0
# Multiply-back construction from Appendix C: https://arxiv.org/pdf/1805.12445#page=9

"""Fixed-point square root from the live C1 inverse-square-root estimate."""

from ...core.ir import Instruction
from .bit_compression import with_weighted_columns_ir
from .bitwise import _validate_wires
from .integer_product import _product_terms, integer_product_requirements
from .inverse_sqrt import _inverse_sqrt_layout, _with_inverse_sqrt_estimate_ir


def fixed_point_sqrt_requirements(
    source_width: int, *, source_fraction_bits: int, precision: int, iterations: int
) -> tuple[int, int]:
    """Return clean history plus shared scratch and reusable measurement slots."""
    estimate_width, (seed_work, seed_slots), (step_work, step_slots) = (
        _inverse_sqrt_layout(source_width, source_fraction_bits, precision, iterations)
    )
    _, product_work = integer_product_requirements(source_width, estimate_width)
    return (
        (iterations + 1) * estimate_width + max(seed_work, step_work, product_work),
        max(seed_slots, step_slots),
    )


def fixed_point_sqrt_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    precision: int,
    iterations: int,
    tuned: bool = False,
) -> tuple[Instruction, ...]:
    """XOR the truncated C1/Newton square root into a big-endian result.

    The result uses ceil(source_integer_bits/2) integer bits. Its remaining
    bits are fractional. The source is preserved, zero has a zero mask and
    clean caller-owned history and scratch are restored. Tuning uses the
    C2 seed and requires at least one Newton update.
    """
    n = len(source)
    f = source_fraction_bits
    estimate_width, (seed_work, seed_slots), (step_work, step_slots) = (
        _inverse_sqrt_layout(n, f, precision, iterations)
    )
    integers = (n - f + 1) // 2
    if len(result) < max(1, integers):
        raise ValueError("Square root result must keep all integer bits")
    _, product_work = integer_product_requirements(n, estimate_width)
    history_width = (iterations + 1) * estimate_width
    work = history_width + max(seed_work, step_work, product_work)
    slots = max(seed_slots, step_slots)
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(f"Square root requires {work} work wires and {slots} slots")
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Square root requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    offset = f + precision - (len(result) - integers)
    drop = max(0, offset)
    target = result[: len(result) + min(0, offset)]
    scratch = workspace[history_width : history_width + product_work]

    def copy_root(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (wire, out))
            for wire, out in zip(
                reversed(word[: len(word) - drop]), reversed(target), strict=False
            )
            if wire is not None
        )

    def multiply(estimate: tuple[int, ...]) -> tuple[Instruction, ...]:
        return with_weighted_columns_ir(
            n + estimate_width,
            scratch,
            classical_bits[0],
            _product_terms(source, estimate),
            copy_root,
        )

    return _with_inverse_sqrt_estimate_ir(
        source,
        workspace,
        classical_bits,
        source_fraction_bits=f,
        precision=precision,
        iterations=iterations,
        tuned=tuned,
        consume=multiply,
    )
