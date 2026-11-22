# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 _cirq_to_bloq._ensure_in_reg_exists.

"""Gather already-owned word views with source-ordered splits, casts and joins."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import RegisterType
from ..component_builder import ComponentProgramBuilder
from .component_cast import cast_word_function


def gather_word_binding(
    builder: ComponentProgramBuilder,
    words: dict[tuple[int, ...], RegisterBinding],
    wires: tuple[int, ...],
    dtype: RegisterType,
) -> RegisterBinding:
    """Consume intersecting views only when the requested word is absent.

    Wire labels identify positions within caller-owned storage. This routine
    performs no allocation. The mapping retains disjoint live words and is
    updated after each split, cast or join. Callers replace a gathered value
    with its component output before requesting another view.
    """
    if wires in words:
        return words[wires]
    requested = set(wires)
    if not requested or not requested <= {wire for word in words for wire in word}:
        raise ValueError("A gathered view requires existing owned wire positions")
    split: dict[tuple[int, ...], RegisterBinding] = {}
    for word, binding in words.items():
        if len(word) > 1 and requested.intersection(word):
            split.update(
                ((wire,), bit)
                for wire, bit in zip(word, builder.split(binding), strict=True)
            )
        else:
            split[word] = binding
    words.clear()
    selected: dict[int, RegisterBinding] = {}
    for word, binding in split.items():
        if not requested.intersection(word):
            words[word] = binding
            continue
        target = quantum_bit_type() if len(wires) > 1 else dtype
        if binding.register.dtype != target:
            function = cast_word_function(
                "word_view_cast", binding.register.dtype, target
            )
            value = builder.call_program(
                ProgramRecord((function,)), {"reg": binding}, entry=function.name
            )["reg"]
            assert isinstance(value, RegisterBinding)
            binding = value
        selected[word[0]] = binding
    words[wires] = (
        builder.join([selected[wire] for wire in wires], dtype)
        if len(wires) > 1
        else selected[wires[0]]
    )
    return words[wires]
