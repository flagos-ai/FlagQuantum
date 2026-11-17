# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix inverse generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_inverse.cc

"""Itoh–Tsujii field inverse and measured cleanup on fixed wires."""

from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ..primitives.measurement import delete_zero_register_ir, reset_x_register_ir
from ._physical_calls import append_field_call
from .binary_linear_map import append_field_frobenius, field_frobenius_ir
from .field_multiplication import _validate_field
from .polynomial import _validate_registers
from .polynomial_product import append_field_product


def field_inverse_chain_size(width: int) -> int:
    """Return the number of caller-owned clean wires for the explicit chain."""
    if type(width) is not int or width < 1:
        raise ValueError("Field inverse width must be a positive integer")
    exponent = width - 1
    if exponent == 0:
        return 0
    return width * (exponent.bit_length() + max(0, exponent.bit_count() - 2) - 1)


def field_inverse_workspace_size(width: int) -> int:
    """Return the clean width for a self-cleaning inverse."""
    chain_size = field_inverse_chain_size(width)
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


def _append_double_word(
    gates: list[Instruction],
    words: tuple[tuple[int, ...], ...],
    i: int,
    scratch: tuple[int, ...],
    modulus: int,
    *,
    bits: tuple[int, ...] | None = None,
    parity_bit: int | None = None,
    calls: list[ComponentCall] | None = None,
) -> None:
    previous = words[i - 1]
    count = 1 << (i - 1)
    append_field_call(gates, calls, "gf_iadd", _word_xor(previous, scratch))
    append_field_frobenius(gates, calls, scratch, count, modulus)
    append_field_product(
        gates,
        calls,
        previous,
        scratch,
        words[i],
        modulus,
        bits=bits,
        parity_bit=parity_bit,
    )
    append_field_frobenius(gates, calls, scratch, count, modulus, adjoint=True)
    append_field_call(gates, calls, "gf_iadd", _word_xor(previous, scratch))


def _build_folds(
    gates: list[Instruction],
    words: tuple[tuple[int, ...], ...],
    start: int,
    folds: tuple[int, ...],
    modulus: int,
    calls: list[ComponentCall] | None = None,
) -> None:
    for i, power in enumerate(folds, start=start):
        left = words[i]
        append_field_frobenius(gates, calls, left, 1 << power, modulus)
        append_field_product(gates, calls, left, words[power], words[i + 1], modulus)


def _erase_folds(
    gates: list[Instruction],
    words: tuple[tuple[int, ...], ...],
    start: int,
    folds: tuple[int, ...],
    bits: tuple[int, ...],
    modulus: int,
    parity_bit: int | None,
    calls: list[ComponentCall] | None = None,
) -> None:
    for i, power in reversed(tuple(enumerate(folds, start=start))):
        left = words[i]
        append_field_product(
            gates,
            calls,
            left,
            words[power],
            words[i + 1],
            modulus,
            bits=bits,
            parity_bit=parity_bit,
        )
        append_field_frobenius(gates, calls, left, 1 << power, modulus, adjoint=True)


def _delete_zero_word(word: tuple[int, ...], bit: int) -> tuple[Instruction, ...]:
    return delete_zero_register_ir(word[::-1], bit)


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


def field_inverse_chain_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    *,
    modulus: int,
    calls: list[ComponentCall] | None = None,
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
        workspace_size=field_inverse_chain_size(len(source)),
    )
    if len(source) == 1:
        gates: list[Instruction] = []
        append_field_call(gates, calls, "gf_iadd", _word_xor(source, target))
        return tuple(gates)
    doubles, folds = _chain_steps(len(source))
    gates = []
    for i in doubles:
        _append_double_word(gates, words, i, target, modulus, calls=calls)
    _build_folds(gates, words, len(doubles), folds, modulus, calls)
    if not folds:
        if doubles:
            gates.extend(_word_swap(words[len(doubles)], target))
        else:
            append_field_call(gates, calls, "gf_iadd", _word_xor(source, target))
    append_field_frobenius(gates, calls, target, 1, modulus)
    return tuple(gates)


def field_inverse_chain_erase_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    chain: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    calls: list[ComponentCall] | None = None,
    parity_bit: int | None = None,
    emit_resets: bool = False,
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
        workspace_size=field_inverse_chain_size(len(source)),
    )
    doubles, folds = _chain_steps(len(source))
    expected = len(source) if doubles or folds else 1
    _validate_inverse_slots(measurement_bits, expected, parity_bit)
    gates: list[Instruction] = []
    append_field_frobenius(gates, calls, target, 1, modulus, adjoint=True)
    if folds:
        _erase_folds(
            gates,
            words,
            len(doubles),
            folds,
            measurement_bits,
            modulus,
            parity_bit,
            calls,
        )
        if emit_resets:
            gates.extend(reset_x_register_ir(target[::-1]))
    else:
        if doubles:
            gates.extend(_word_swap(words[len(doubles)], target))
        else:
            append_field_call(gates, calls, "gf_iadd", _word_xor(source, target))
    for i in reversed(doubles):
        _append_double_word(
            gates,
            words,
            i,
            target,
            modulus,
            bits=measurement_bits,
            parity_bit=parity_bit,
            calls=calls,
        )
    gates.extend(_delete_zero_word(target, measurement_bits[-1]))
    return tuple(gates)


def field_inverse_ir(
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
        workspace_size=field_inverse_workspace_size(width),
    )
    doubles, folds = _chain_steps(width)
    expected = width if folds or len(doubles) > 1 else len(doubles)
    _validate_inverse_slots(measurement_bits, expected, parity_bit)
    gates: list[Instruction] = []
    for i in doubles:
        _append_double_word(gates, words, i, target, modulus)
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
        _append_double_word(
            gates,
            words,
            i,
            scratch,
            modulus,
            bits=measurement_bits,
            parity_bit=parity_bit,
        )
    if scratch:
        gates.extend(_delete_zero_word(scratch, measurement_bits[-1]))
    gates.extend(field_frobenius_ir(target, modulus=modulus))
    return tuple(gates)


def field_inverse_erase_ir(
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
        workspace_size=field_inverse_workspace_size(width),
    )
    doubles, folds = _chain_steps(width)
    _validate_inverse_slots(measurement_bits, width if doubles else 1, parity_bit)
    gates = list(field_frobenius_ir(target, modulus=modulus, adjoint=True))
    for i in doubles if folds else doubles[:-1]:
        _append_double_word(gates, words, i, words[i + 1], modulus)
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
        _append_double_word(
            gates,
            words,
            i,
            target,
            modulus,
            bits=measurement_bits,
            parity_bit=parity_bit,
        )
    gates.extend(_delete_zero_word(target, measurement_bits[-1]))
    return tuple(gates)
