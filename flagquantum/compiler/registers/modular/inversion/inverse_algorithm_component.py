# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiModInverseImpl.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose initialization, shared inverse iterations and zero-sentinel cleanup."""

import numpy as np
from numpy.typing import NDArray

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_bindings import RegisterBinding
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.registers import RegisterType
from .....core.ir.program.signatures import Signature
from .....core.value_math import SymbolicInt
from ...arithmetic.constant_addition_component import constant_addition_program
from ...bits.bitwise_component import bitwise_word_program
from ...bits.bitwise_component_control import controlled_bitwise_word_program
from ...comparison._component_gates import comparison_bit_functions
from ...component_builder import BindingCollection, ComponentProgramBuilder
from .inverse_iteration_component import inverse_iteration_program


def _initialize(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    modulus_load: ProgramRecord,
    unit_load: ProgramRecord,
    adjoint: bool,
) -> None:
    if not adjoint:
        values["f"] = builder.call_dict("comparison_x", {"q": values["f"]})["q"]
    for target, loader in (
        (("s", unit_load), ("u", modulus_load))
        if adjoint
        else (("u", modulus_load), ("s", unit_load))
    ):
        values[target] = builder.call_program(
            loader, {"x": values[target]}, entry=loader.functions[0].name
        )["x"]
    if adjoint:
        values["f"] = builder.call_dict("comparison_x", {"q": values["f"]})["q"]


def _correct(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    terminals: NDArray[np.object_],
    modulus_load: ProgramRecord,
    unit_load: ProgramRecord,
    complement: ProgramRecord,
    addition: ProgramRecord,
    controlled_modulus: ProgramRecord,
    controlled_predecessor: ProgramRecord,
    adjoint: bool,
) -> None:
    def call_word(record: ProgramRecord, target: str) -> None:
        values[target] = builder.call_program(
            record, {"x": values[target]}, entry=record.functions[0].name
        )["x"]

    def controlled(target: str, record: ProgramRecord) -> None:
        result = builder.call_program(
            record,
            {"ctrl": terminals[0], "x": values[target]},
            entry=record.functions[0].name,
        )
        terminals[0], values[target] = result["ctrl"], result["x"]

    if not adjoint:
        call_word(complement, "r")
        call_word(addition, "r")
    if adjoint:
        for target, loader in (
            ("s", controlled_predecessor),
            ("u", controlled_predecessor),
            ("r", controlled_modulus),
        ):
            controlled(target, loader)
    for target, loader in (
        (("s", modulus_load), ("u", unit_load))
        if adjoint
        else (("u", unit_load), ("s", modulus_load))
    ):
        call_word(loader, target)
    if not adjoint:
        for target, loader in (
            ("r", controlled_modulus),
            ("u", controlled_predecessor),
            ("s", controlled_predecessor),
        ):
            controlled(target, loader)
    if adjoint:
        call_word(addition, "r")
        call_word(complement, "r")


def inverse_algorithm_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain the two 2n-bit logical histories across 2n shared iteration calls.

    Four words are MSB-first Montgomery values. Histories are opaque MSB words.
    Clean initialization and source-reachable or matching inverse inputs are
    required. The source all-ones modulus p+1 loader failure remains explicit.
    """
    word = montgomery_register_type(width, modulus)
    iteration = inverse_iteration_program(width, modulus, adjoint=adjoint)
    return inverse_algorithm_component_program(
        width,
        modulus,
        iteration,
        bitwise_word_program(word, "xor_constant", pattern=modulus),
        bitwise_word_program(word, "xor_constant", pattern=1),
        bitwise_word_program(word, "bitwise_not"),
        constant_addition_program(word, modulus + 1, adjoint=adjoint),
        controlled_bitwise_word_program(word, "xor_constant", pattern=modulus),
        controlled_bitwise_word_program(word, "xor_constant", pattern=modulus - 1),
        adjoint=adjoint,
    )


def inverse_algorithm_component_program(
    width: int,
    modulus: SymbolicInt,
    iteration: ProgramRecord,
    modulus_load: ProgramRecord,
    unit_load: ProgramRecord,
    complement: ProgramRecord,
    addition: ProgramRecord,
    controlled_modulus: ProgramRecord,
    controlled_predecessor: ProgramRecord,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain the source algorithm around explicit constant and iteration owners.

    Only genuinely unavailable children may remain declarations. Initialized
    working words and caller-owned histories follow source order; unbound
    parameters provide no executable cleanup or resource assertion.
    """
    word, bit = montgomery_register_type(width, modulus), quantum_bit_type()
    history = RegisterType("opaque", 2 * width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            u=word, v=word, r=word, s=word, m=history, f=bit, terminal_condition=history
        ),
        comparison_bit_functions(),
    )
    values = dict(builder.inputs)
    if not adjoint:
        _initialize(builder, values, modulus_load, unit_load, adjoint)
    m, terminal = values.pop("m"), values.pop("terminal_condition")
    assert isinstance(m, RegisterBinding) and isinstance(terminal, RegisterBinding)
    markers, terminals = builder.split(m), builder.split(terminal)
    if adjoint:
        _correct(
            builder,
            values,
            terminals,
            modulus_load,
            unit_load,
            complement,
            addition,
            controlled_modulus,
            controlled_predecessor,
            adjoint,
        )
    for i in reversed(range(2 * width)) if adjoint else range(2 * width):
        result = builder.call_program(
            iteration, {**values, "m": markers[i], "is_terminal": terminals[i]}
        )
        markers[i], terminals[i] = result.pop("m"), result.pop("is_terminal")
        values.update(result)
    if not adjoint:
        _correct(
            builder,
            values,
            terminals,
            modulus_load,
            unit_load,
            complement,
            addition,
            controlled_modulus,
            controlled_predecessor,
            adjoint,
        )
    values["m"], values["terminal_condition"] = builder.join(markers), builder.join(
        terminals
    )
    if adjoint:
        _initialize(builder, values, modulus_load, unit_load, adjoint)
    return builder.finalize(values)
