# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 _cirq_to_bloq._ensure_in_reg_exists.

"""Gather already-owned word views with source-ordered splits, casts and joins."""

from collections.abc import Mapping

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .component_cast import cast_word_function


def gather_word_binding(
    builder: ComponentProgramBuilder,
    words: dict[tuple[int, ...], RegisterBinding],
    wires: tuple[int, ...],
    dtype: RegisterType,
    *,
    nominal_types: dict[tuple[int, ...], RegisterType] | None = None,
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
    types = (
        nominal_types
        if nominal_types is not None
        else {word: binding.register.dtype for word, binding in words.items()}
    )
    split: dict[tuple[int, ...], RegisterBinding] = {}
    split_types: dict[tuple[int, ...], RegisterType] = {}
    for word, binding in words.items():
        if len(word) > 1 and requested.intersection(word):
            split.update(
                ((wire,), bit)
                for wire, bit in zip(word, builder.split(binding), strict=True)
            )
            split_types.update(((wire,), quantum_bit_type()) for wire in word)
        else:
            split[word] = binding
            split_types[word] = types[word]
    words.clear()
    types.clear()
    selected: dict[int, RegisterBinding] = {}
    for word, binding in split.items():
        if not requested.intersection(word):
            words[word] = binding
            types[word] = split_types[word]
            continue
        target = quantum_bit_type() if len(wires) > 1 else dtype
        if split_types[word] != target:
            function = cast_word_function("word_view_cast", split_types[word], target)
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
    types[wires] = dtype
    return words[wires]


def call_through_word_program(
    builder: ComponentProgramBuilder,
    words: dict[tuple[int, ...], RegisterBinding],
    record: ProgramRecord,
    positions: Mapping[str, tuple[int, ...]],
    *,
    nominal_types: dict[tuple[int, ...], RegisterType] | None = None,
) -> None:
    """Apply a scalar THRU-word call and replace only its consumed word views.

    Fourier compositions use this shared operation for repeated prefix and
    reversed-slice calls. The component owns its private work. Shaped or sided
    ports require their existing explicit ownership path.
    """
    signature = Signature(record.functions[0].registers or ())
    if set(positions) != {port.name for port in signature} or any(
        port.shape_symbolic or port.side != RegisterSide.THRU for port in signature
    ):
        raise ValueError("Word-view calls require matching scalar through ports")
    inputs = {
        port.name: gather_word_binding(
            builder,
            words,
            positions[port.name],
            port.dtype,
            nominal_types=nominal_types,
        )
        for port in signature
    }
    returned = builder.call_program(record, inputs, entry=record.functions[0].name)
    for port in signature:
        value = returned[port.name]
        assert isinstance(value, RegisterBinding)
        words[positions[port.name]] = value
        if nominal_types is not None:
            # Python dict assignment retains the original equal _QReg key.
            # Its nominal dtype can differ from the latest returned handle.
            nominal_types.setdefault(positions[port.name], port.dtype)
