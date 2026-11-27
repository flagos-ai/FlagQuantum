# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 controlled and half linear comparisons.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Shared sign extension, complement and cast sequence for linear comparisons."""

from dataclasses import replace

from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import unsigned_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ..bits.bitwise_component import bitwise_word_program
from ..bits.sign_extension_component import sign_extension_program
from ..component_builder import ComponentProgramBuilder
from ..layout.component_cast import cast_word_function


def comparison_width(dtype: RegisterType, *, allow_empty: bool = False) -> int:
    """Require the concrete integer encodings admitted by the source family."""
    width = executable_register_width(dtype)
    if allow_empty and width == 0 and dtype.encoding in {"uint", "montgomery"}:
        return width
    if width < 1 or dtype.encoding not in {"uint", "signed", "montgomery"}:
        raise ValueError("Linear comparison requires a positive concrete integer dtype")
    return width


def prepare_difference_words(
    builder: ComponentProgramBuilder,
    dtype: RegisterType,
    a: RegisterBinding,
    b: RegisterBinding,
) -> tuple[RegisterBinding, RegisterBinding]:
    """Widen a/b, complement b and expose unsigned adder views in source order."""
    width = comparison_width(dtype, allow_empty=True)
    wide = replace(dtype, quantum_bits=width + 1)
    words: list[RegisterBinding] = []
    for word in (a, b):
        if dtype.encoding == "signed":
            result = builder.call_program(
                sign_extension_program(dtype, wide), {"x": word}
            )["y"]
            assert isinstance(result, RegisterBinding)
            word = result
        else:
            prefix = builder.allocate()
            word = builder.join([prefix, *builder.split(word)])
        words.append(word)
    result = builder.call_program(
        bitwise_word_program(wide, "bitwise_not"), {"x": words[1]}
    )["x"]
    assert isinstance(result, RegisterBinding)
    words[1] = result
    cast = cast_word_function(
        "difference_unsigned", wide, unsigned_register_type(width + 1)
    )
    builder.define(cast)
    for i, word in enumerate(words):
        result = builder.call_dict(cast.name, {"reg": word})["reg"]
        assert isinstance(result, RegisterBinding)
        words[i] = result
    return words[0], words[1]


def restore_difference_words(
    builder: ComponentProgramBuilder,
    dtype: RegisterType,
    a: RegisterBinding,
    b: RegisterBinding,
) -> tuple[RegisterBinding, RegisterBinding]:
    """Restore casts and b, then clear and free the two redundant prefixes."""
    width = comparison_width(dtype, allow_empty=True)
    wide = replace(dtype, quantum_bits=width + 1)
    cast = cast_word_function(
        "difference_original", unsigned_register_type(width + 1), wide
    )
    builder.define(cast)
    words: list[RegisterBinding] = []
    for word in (a, b):
        result = builder.call_dict(cast.name, {"reg": word})["reg"]
        assert isinstance(result, RegisterBinding)
        words.append(result)
    result = builder.call_program(
        bitwise_word_program(wide, "bitwise_not"), {"x": words[1]}
    )["x"]
    assert isinstance(result, RegisterBinding)
    words[1] = result
    prefixes: list[RegisterBinding] = []
    for i, word in enumerate(words):
        if dtype.encoding == "signed":
            result = builder.call_program(
                sign_extension_program(dtype, wide, adjoint=True), {"x": word}
            )["y"]
            assert isinstance(result, RegisterBinding)
            words[i] = result
        else:
            bits = builder.split(word)
            words[i] = builder.join(bits[1:])
            prefixes.append(bits[0])
    for prefix in prefixes:
        builder.free(prefix)
    return words[0], words[1]
