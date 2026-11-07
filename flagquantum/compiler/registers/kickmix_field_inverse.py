# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix inverse generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_inverse.cc

"""Itoh–Tsujii field inverse and measured cleanup on fixed wires."""

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


def kickmix_field_inverse_workspace_size(width: int) -> int:
    """Return the clean width for a self-cleaning inverse."""
    chain_size = kickmix_field_inverse_chain_size(width)
    return chain_size + (width if (width - 1).bit_count() > 1 else 0)


def _chain_words(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    modulus: int,
    *,
    workspace_size: int,
) -> tuple[tuple[int, ...], ...]:
    width = len(source)
    _validate_field(width, modulus)
    _validate_registers(source, target, chain)
    if len(target) != width or len(chain) != workspace_size:
        raise ValueError("Field inverse target or workspace has the wrong width")
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


def _double_word(
    words: tuple[tuple[int, ...], ...], i: int, scratch: tuple[int, ...], modulus: int
) -> tuple[Instruction, ...]:
    previous = words[i - 1]
    count = 1 << (i - 1)
    return (
        *_word_xor(previous, scratch),
        *kickmix_field_frobenius_ir(scratch, count, modulus=modulus),
        *kickmix_field_product_xor_ir(previous, scratch, words[i], modulus=modulus),
        *kickmix_field_frobenius_ir(scratch, count, modulus=modulus, adjoint=True),
        *_word_xor(previous, scratch),
    )


def _erase_double_word(
    words: tuple[tuple[int, ...], ...],
    i: int,
    scratch: tuple[int, ...],
    bits: tuple[int, ...],
    modulus: int,
    parity_bit: int | None,
) -> tuple[Instruction, ...]:
    previous = words[i - 1]
    count = 1 << (i - 1)
    return (
        *_word_xor(previous, scratch),
        *kickmix_field_frobenius_ir(scratch, count, modulus=modulus),
        *kickmix_field_product_erase_ir(
            previous, scratch, words[i], bits, modulus=modulus, parity_bit=parity_bit
        ),
        *kickmix_field_frobenius_ir(scratch, count, modulus=modulus, adjoint=True),
        *_word_xor(previous, scratch),
    )


def _build_folds(
    gates: list[Instruction],
    words: tuple[tuple[int, ...], ...],
    start: int,
    folds: tuple[int, ...],
    modulus: int,
) -> None:
    for i, power in enumerate(folds, start=start):
        left = words[i]
        gates.extend(kickmix_field_frobenius_ir(left, 1 << power, modulus=modulus))
        gates.extend(
            kickmix_field_product_xor_ir(
                left, words[power], words[i + 1], modulus=modulus
            )
        )


def _erase_folds(
    gates: list[Instruction],
    words: tuple[tuple[int, ...], ...],
    start: int,
    folds: tuple[int, ...],
    bits: tuple[int, ...],
    modulus: int,
    parity_bit: int | None,
) -> None:
    for i, power in reversed(tuple(enumerate(folds, start=start))):
        left = words[i]
        gates.extend(
            kickmix_field_product_erase_ir(
                left,
                words[power],
                words[i + 1],
                bits,
                modulus=modulus,
                parity_bit=parity_bit,
            )
        )
        gates.extend(
            kickmix_field_frobenius_ir(left, 1 << power, modulus=modulus, adjoint=True)
        )


def _delete_zero_word(word: tuple[int, ...], bit: int) -> tuple[Instruction, ...]:
    return tuple(gate for wire in word[::-1] for gate in measure_x_clear_ir(wire, bit))


def _validate_inverse_slots(
    measurement_bits: tuple[int, ...], expected: int, parity_bit: int | None
) -> None:
    slots = measurement_bits if parity_bit is None else (*measurement_bits, parity_bit)
    if (
        len(measurement_bits) != expected
        or len(set(slots)) != len(slots)
        or any(type(bit) is not int or bit < 0 for bit in slots)
    ):
        raise ValueError(
            "Field inverse cleanup requires distinct nonnegative measurement slots"
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
    words = _chain_words(
        source,
        target,
        chain,
        modulus,
        workspace_size=kickmix_field_inverse_chain_size(len(source)),
    )
    if len(source) == 1:
        return _word_xor(source, target)
    doubles, folds = _chain_steps(len(source))
    gates: list[Instruction] = []
    for i in doubles:
        gates.extend(_double_word(words, i, target, modulus))
    _build_folds(gates, words, len(doubles), folds, modulus)
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
    words = _chain_words(
        source,
        target,
        chain,
        modulus,
        workspace_size=kickmix_field_inverse_chain_size(len(source)),
    )
    doubles, folds = _chain_steps(len(source))
    expected = len(source) if doubles or folds else 1
    _validate_inverse_slots(measurement_bits, expected, parity_bit)
    gates = list(kickmix_field_frobenius_ir(target, modulus=modulus, adjoint=True))
    if folds:
        _erase_folds(
            gates, words, len(doubles), folds, measurement_bits, modulus, parity_bit
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
        gates.extend(
            _erase_double_word(words, i, target, measurement_bits, modulus, parity_bit)
        )
    gates.extend(_delete_zero_word(target, measurement_bits[-1]))
    return tuple(gates)


def kickmix_field_inverse_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Write a big-endian inverse into a clean target and clear the workspace.

    Zero maps to zero. Supply one outcome slot at width three, no slots at
    widths one and two, and one per field bit otherwise.
    """
    width = len(source)
    words = _chain_words(
        source,
        target,
        workspace,
        modulus,
        workspace_size=kickmix_field_inverse_workspace_size(width),
    )
    doubles, folds = _chain_steps(width)
    expected = width if folds or len(doubles) > 1 else len(doubles)
    _validate_inverse_slots(measurement_bits, expected, parity_bit)
    gates: list[Instruction] = []
    for i in doubles:
        gates.extend(_double_word(words, i, target, modulus))
    _build_folds(gates, words, len(doubles), folds, modulus)

    if folds:
        gates.extend(_word_xor(words[-2], target))
        _erase_folds(
            gates, words, len(doubles), folds, measurement_bits, modulus, parity_bit
        )
        # The first folded result is physically zero after HMR and can be
        # reused directly as scratch for the doubling cleanup.
        scratch = words[len(doubles) + 1]
        cleanup = doubles
    elif doubles:
        scratch = words[len(doubles)]
        gates.extend(_word_swap(scratch, target))
        cleanup = doubles[:-1]
    else:
        scratch = ()
        cleanup = ()
        gates.extend(_word_xor(source, target))
    for i in reversed(cleanup):
        gates.extend(
            _erase_double_word(words, i, scratch, measurement_bits, modulus, parity_bit)
        )
    if scratch:
        gates.extend(_delete_zero_word(scratch, measurement_bits[-1]))
    gates.extend(kickmix_field_frobenius_ir(target, modulus=modulus))
    return tuple(gates)


def kickmix_field_inverse_erase_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Rebuild the inverse chain and erase a known inverse in the target.

    The workspace starts and finishes clean. Supply one result slot for
    widths one and two, and one per field bit otherwise.
    """
    width = len(source)
    words = _chain_words(
        source,
        target,
        workspace,
        modulus,
        workspace_size=kickmix_field_inverse_workspace_size(width),
    )
    doubles, folds = _chain_steps(width)
    _validate_inverse_slots(measurement_bits, width if doubles else 1, parity_bit)
    gates = list(kickmix_field_frobenius_ir(target, modulus=modulus, adjoint=True))
    for i in doubles if folds else doubles[:-1]:
        gates.extend(_double_word(words, i, words[i + 1], modulus))
    if folds:
        _build_folds(gates, words, len(doubles), folds, modulus)
        gates.extend(_word_xor(words[-2], target))
        _erase_folds(
            gates, words, len(doubles), folds, measurement_bits, modulus, parity_bit
        )
    elif doubles:
        gates.extend(_word_swap(words[len(doubles)], target))
    else:
        gates.extend(_word_xor(source, target))
    for i in reversed(doubles):
        gates.extend(
            _erase_double_word(words, i, target, measurement_bits, modulus, parity_bit)
        )
    gates.extend(_delete_zero_word(target, measurement_bits[-1]))
    return tuple(gates)
