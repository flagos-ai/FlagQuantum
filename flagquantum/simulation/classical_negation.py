# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Negate.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/negate.py

"""Classical negation through the existing complement and constant-add models."""

from .classical_addition import add_constant_classical_value
from .classical_bitwise import IntegerValue, bitwise_not_classical_value
from .integer_values import validate_integer_word_value


def negate_classical_value(
    target: IntegerValue, width: int, *, signed: bool = False
) -> int:
    """Compose BitwiseNot and AddK(1) on a concrete positive-width word.

    The source zero-width composite cannot bind x and raises TypeError; keep
    that failure explicit. Symbolic widths remain unsupported by the AddK
    consumer. This value model does not repair upstream gate decomposition
    failures such as the one-bit signed constant-load boundary.
    """
    if width == 0:
        raise TypeError("Source zero-width negation cannot bind its input word")
    validate_integer_word_value(target, width, signed=signed)
    complement = bitwise_not_classical_value(target, width, unsigned=not signed)
    validate_integer_word_value(complement, width, signed=signed)
    result = add_constant_classical_value(complement, 1, width, signed=signed)
    validate_integer_word_value(result, width, signed=signed)
    return result
