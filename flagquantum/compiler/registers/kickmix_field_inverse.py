# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix explicit-chain inverse generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_inverse.cc

"""Explicit Itoh–Tsujii field inverse and measured cleanup on fixed wires."""

from ...core.ir import Instruction
from .field_multiplication import _validate_field
from .kickmix_field_erasure import kickmix_field_product_erase_ir
from .kickmix_field_linear import kickmix_field_frobenius_ir
from .kickmix_field_product import kickmix_field_product_xor_ir
from .measurement import measure_x_clear_ir
from .polynomial import _validate_registers


def kickmix_field_inverse_chain_size(width: int) -> int:
    """Return the number of caller-owned clean wires for the explicit chain."""
    if type(width) is not int or width < 1:
        raise ValueError("Field inverse width must be a positive integer")
    exponent = width - 1
    if exponent == 0:
        return 0
    return width * (exponent.bit_length() + max(0, exponent.bit_count() - 2) - 1)


def _chain_words(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    modulus: int,
) -> tuple[tuple[int, ...], ...]:
    width = len(source)
    _validate_field(width, modulus)
    _validate_registers(source, target, chain)
    if len(target) != width or len(chain) != kickmix_field_inverse_chain_size(width):
        raise ValueError("Field inverse target or explicit chain has the wrong width")
    return (
        source,
        *(chain[i : i + width] for i in range(0, len(chain), width)),
        target,
    )


def _chain_steps(width: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    exponent = width - 1
    set_bits = tuple(
        i for i in reversed(range(exponent.bit_length())) if exponent >> i & 1
    )
    return tuple(range(1, exponent.bit_length())), set_bits[1:]


def _word_xor(
    source: tuple[int, ...], target: tuple[int, ...]
) -> tuple[Instruction, ...]:
    return tuple(
        Instruction("cx", wires)
        for wires in zip(source[::-1], target[::-1], strict=True)
    )


def _word_swap(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    return tuple(
        Instruction("swap", wires)
        for wires in zip(left[::-1], right[::-1], strict=True)
    )


def kickmix_field_inverse_chain_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    *,
    modulus: int,
) -> tuple[Instruction, ...]:
    """Compute the inverse of a big-endian source, mapping zero to zero.

    Target and each consecutive big-endian chain word start clean. Source is
    preserved, and the chain stays live for its measured cleanup partner.
    """
    words = _chain_words(source, target, chain, modulus)
    if len(source) == 1:
        return _word_xor(source, target)
    doubles, folds = _chain_steps(len(source))
    gates: list[Instruction] = []
    for i in doubles:
        previous = words[i - 1]
        count = 1 << (i - 1)
        gates.extend(_word_xor(previous, target))
        gates.extend(kickmix_field_frobenius_ir(target, count, modulus=modulus))
        gates.extend(
            kickmix_field_product_xor_ir(previous, target, words[i], modulus=modulus)
        )
        gates.extend(
            kickmix_field_frobenius_ir(target, count, modulus=modulus, adjoint=True)
        )
        gates.extend(_word_xor(previous, target))
    for i, power in enumerate(folds, start=len(doubles)):
        left = words[i]
        gates.extend(kickmix_field_frobenius_ir(left, 1 << power, modulus=modulus))
        gates.extend(
            kickmix_field_product_xor_ir(
                left, words[power], words[i + 1], modulus=modulus
            )
        )
    if not folds:
        gates.extend(
            _word_swap(words[len(doubles)], target)
            if doubles
            else _word_xor(source, target)
        )
    gates.extend(kickmix_field_frobenius_ir(target, modulus=modulus))
    return tuple(gates)


def kickmix_field_inverse_chain_erase_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Erase the known inverse and its live chain, preserving the source.

    Measurement slots are caller-owned and reused across products. The final
    deletion of known-zero target wires reuses the least-significant slot.
    """
    words = _chain_words(source, target, chain, modulus)
    doubles, folds = _chain_steps(len(source))
    expected = len(source) if doubles or folds else 1
    slots = measurement_bits if parity_bit is None else (*measurement_bits, parity_bit)
    if (
        len(measurement_bits) != expected
        or len(set(slots)) != len(slots)
        or any(type(bit) is not int or bit < 0 for bit in slots)
    ):
        raise ValueError(
            "Field inverse cleanup requires distinct nonnegative measurement slots"
        )
    gates = list(kickmix_field_frobenius_ir(target, modulus=modulus, adjoint=True))
    if folds:
        for i, power in reversed(tuple(enumerate(folds, start=len(doubles)))):
            left = words[i]
            gates.extend(
                kickmix_field_product_erase_ir(
                    left,
                    words[power],
                    words[i + 1],
                    measurement_bits,
                    modulus=modulus,
                    parity_bit=parity_bit,
                )
            )
            gates.extend(
                kickmix_field_frobenius_ir(
                    left, 1 << power, modulus=modulus, adjoint=True
                )
            )
        # The fold's HMR already returns the fixed physical target wires to
        # zero. The source resets them to reactivate their logical lifetime.
    else:
        gates.extend(
            _word_swap(words[len(doubles)], target)
            if doubles
            else _word_xor(source, target)
        )
    for i in reversed(doubles):
        previous = words[i - 1]
        count = 1 << (i - 1)
        gates.extend(_word_xor(previous, target))
        gates.extend(kickmix_field_frobenius_ir(target, count, modulus=modulus))
        gates.extend(
            kickmix_field_product_erase_ir(
                previous,
                target,
                words[i],
                measurement_bits,
                modulus=modulus,
                parity_bit=parity_bit,
            )
        )
        gates.extend(
            kickmix_field_frobenius_ir(target, count, modulus=modulus, adjoint=True)
        )
        gates.extend(_word_xor(previous, target))
    for wire in target[::-1]:
        gates.extend(measure_x_clear_ir(wire, measurement_bits[-1]))
    return tuple(gates)
