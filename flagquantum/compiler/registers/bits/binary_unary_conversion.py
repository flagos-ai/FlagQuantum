# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_binary_to_unary and gen_unary_to_binary.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/lookup/gen_lookup.cc

"""Source-ordered in-place binary/unary conversion and measured erasure."""

from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ..finite_field.polynomial import _validate_registers


def _source_word(target: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    _validate_registers(target)
    size = len(target)
    if not size or size & (size - 1):
        raise ValueError(
            "Binary/unary conversion requires a power-of-two target length"
        )
    return target[::-1], size.bit_length() - 1


def binary_to_unary_operations(target: tuple[int, ...]) -> tuple[Operation, ...]:
    """Convert the low binary subword into a one-hot word, resetting other ports."""
    word, bits = _source_word(target)
    operations = [Operation("r", (wire,)) for wire in word[bits:]]
    for k in reversed(range(bits)):
        operations.append(Operation("swap", (word[k], word[(2 << k) - 1])))
    operations.append(Operation("x", (word[0],)))
    for k in range(bits):
        m = 1 << k
        control = 2 * m - 1
        for j in range(m - 1):
            operations.append(Operation("cswap", (word[control], word[j], word[j + m])))
        for j in range(m - 1):
            operations.append(Operation("cx", (word[j + m], word[control])))
        operations.append(Operation("cx", (word[control], word[m - 1])))
    return tuple(operations)


def unary_to_binary_operations(
    target: tuple[int, ...], *, measurement_bit: int = 0
) -> tuple[Operation, ...]:
    """Erase a valid one-hot buffer into its binary subword with source feedback."""
    word, bits = _source_word(target)
    if type(measurement_bit) is not int or measurement_bit < 0:
        raise ValueError("Unary erasure requires a nonnegative result slot")
    operations = [Operation("swap", (word[k], word[(2 << k) - 1])) for k in range(bits)]
    swapped = list(word)
    for k in reversed(range(bits)):
        index = (2 << k) - 1
        swapped[k], swapped[index] = swapped[index], swapped[k]
    for k in reversed(range(bits)):
        m = 1 << k
        control = 2 * m - 1
        operations.append(Operation("cx", (swapped[control], swapped[m - 1])))
        for j in reversed(range(m - 1)):
            operations.append(Operation("cx", (swapped[j + m], swapped[control])))
        for j in reversed(range(m - 1)):
            operations.extend(
                (
                    Operation("cx", (swapped[j + m], swapped[j])),
                    Operation("hmr", (swapped[j + m],), result_bit=measurement_bit),
                    Operation(
                        "cz",
                        (swapped[control], swapped[j]),
                        condition_bit=measurement_bit,
                    ),
                )
            )
    operations.extend(
        (
            Operation("hmr", (swapped[0],), result_bit=measurement_bit),
            Operation("neg", (), condition_bit=measurement_bit),
        )
    )
    return tuple(operations)
