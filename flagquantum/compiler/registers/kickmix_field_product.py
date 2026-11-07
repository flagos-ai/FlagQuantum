# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix Karatsuba generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_mul.cc

"""Fixed-wire product accumulation with the pinned Kickmix Karatsuba schedule."""

from collections.abc import Iterator

from ...core.ir import Instruction
from .field_multiplication import _field_power, _validate_field
from .finite_field import gf2_add_ir
from .kickmix_field_erasure import kickmix_field_product_erase_ir
from .kickmix_field_linear import (
    kickmix_field_constant_scale_ir,
    kickmix_field_monomial_scale_ir,
)
from .polynomial import _validate_registers


def _scaled_polynomial_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...], shift: int
) -> Iterator[Instruction]:
    """Add (1 + x**shift) times a product to an overlapping target window."""
    if len(left) == 1:
        copy = Instruction("cx", (target[shift], target[0]))
        yield copy
        yield Instruction("ccx", (left[0], right[0], target[shift]))
        yield copy
        return

    product_width = 2 * len(left) - 1
    overlap = max(0, product_width - shift)
    copied = min(shift, product_width)
    # The upper ladder can share wires between CNOTs. Apply it high to low
    # before accumulation and in reverse order afterward.
    upper = tuple(
        Instruction("cx", (target[2 * shift + i], target[shift + i]))
        for i in range(overlap)
    )
    lower = gf2_add_ir(target[shift : shift + copied], target[:copied])
    yield from reversed(upper)
    yield from lower
    yield from _polynomial_product(target[shift : shift + product_width], left, right)
    yield from lower
    yield from upper


def _polynomial_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...]
) -> Iterator[Instruction]:
    """Add a carryless polynomial product on little-endian wires."""
    width = len(left)
    if width == 1:
        yield Instruction("ccx", (left[0], right[0], target[0]))
        return

    half = (width + 1) // 2
    yield from _scaled_polynomial_product(
        target[: 3 * half - 1], left[:half], right[:half], half
    )
    yield from _scaled_polynomial_product(
        target[half:], left[half:], right[half:], half
    )
    yield from _middle_product(target[half : 3 * half - 1], left, right, half)


def _middle_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...], half: int
) -> Iterator[Instruction]:
    low_width = len(left) - half
    add_left = gf2_add_ir(left[half:], left[:low_width])
    add_right = gf2_add_ir(right[half:], right[:low_width])
    yield from add_left
    yield from add_right
    yield from _polynomial_product(target, left[:half], right[:half])
    yield from add_right
    yield from add_left


def kickmix_field_product_xor_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    *,
    modulus: int,
) -> tuple[Instruction, ...]:
    """Add left*right into an arbitrary target over GF(2^m), on fixed wires.

    Registers are disjoint, equally wide, and big-endian. The two factors
    are preserved and no quantum or classical workspace is needed.
    """
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(left, right, target)
    if len(left) != width or len(right) != width:
        raise ValueError("Field product requires equal register widths")
    if width == 1:
        return (Instruction("ccx", (left[0], right[0], target[0])),)

    half = (width + 1) // 2
    high_width = width - half
    factor = 1 | (1 << half)
    inverse = _field_power(factor, (1 << width) - 2, modulus)
    x, y, result = left[::-1], right[::-1], target[::-1]
    return (
        *kickmix_field_monomial_scale_ir(target, -half, modulus=modulus),
        *_middle_product(result[: 2 * half - 1], x, y, half),
        *kickmix_field_constant_scale_ir(target, inverse, modulus=modulus),
        *_polynomial_product(result[: 2 * high_width - 1], x[half:], y[half:]),
        *kickmix_field_monomial_scale_ir(target, half, modulus=modulus),
        *_polynomial_product(result[: 2 * half - 1], x[:half], y[:half]),
        *kickmix_field_constant_scale_ir(target, factor, modulus=modulus),
    )


def kickmix_controlled_field_product_xor_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """XOR the product into target under a positive quantum control.

    Big-endian workspace starts clean and returns to zero on every branch.
    The caller provides one result slot per field bit and a separate parity
    slot when the modulus needs one. The same schedule is its XOR inverse.
    """
    _validate_registers((control,), left, right, target, workspace)
    add = gf2_add_ir(workspace[::-1], target[::-1], control=control)
    return (
        *kickmix_field_product_xor_ir(left, right, workspace, modulus=modulus),
        *add,
        *kickmix_field_product_erase_ir(
            left,
            right,
            workspace,
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )
