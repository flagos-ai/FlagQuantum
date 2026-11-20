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
from ...arithmetic.constant_addition_component import constant_addition_program
from ...bits.bitwise_component import bitwise_word_program
from ...bits.bitwise_component_control import controlled_bitwise_word_program
from ...comparison._component_gates import comparison_bit_functions
from ...component_builder import BindingCollection, ComponentProgramBuilder
from .inverse_iteration_component import inverse_iteration_program


def _initialize(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    word: RegisterType,
    modulus: int,
    adjoint: bool,
) -> None:
    if not adjoint:
        values["f"] = builder.call_dict("comparison_x", {"q": values["f"]})["q"]
    for target, constant in (
        (("s", 1), ("u", modulus)) if adjoint else (("u", modulus), ("s", 1))
    ):
        values[target] = builder.call_program(
            bitwise_word_program(word, "xor_constant", pattern=constant),
            {"x": values[target]},
        )["x"]
    if adjoint:
        values["f"] = builder.call_dict("comparison_x", {"q": values["f"]})["q"]


def _correct(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    terminals: NDArray[np.object_],
    word: RegisterType,
    modulus: int,
    adjoint: bool,
) -> None:
    complement = bitwise_word_program(word, "bitwise_not")
    addition = constant_addition_program(word, modulus + 1, adjoint=adjoint)
    if not adjoint:
        values["r"] = builder.call_program(complement, {"x": values["r"]})["x"]
        values["r"] = builder.call_program(addition, {"x": values["r"]})["x"]
    if adjoint:
        for target, constant in (
            ("s", modulus - 1),
            ("u", modulus - 1),
            ("r", modulus),
        ):
            result = builder.call_program(
                controlled_bitwise_word_program(word, "xor_constant", pattern=constant),
                {"ctrl": terminals[0], "x": values[target]},
            )
            terminals[0], values[target] = result["ctrl"], result["x"]
    for target, constant in (
        (("s", modulus), ("u", 1)) if adjoint else (("u", 1), ("s", modulus))
    ):
        values[target] = builder.call_program(
            bitwise_word_program(word, "xor_constant", pattern=constant),
            {"x": values[target]},
        )["x"]
    if not adjoint:
        for target, constant in (
            ("r", modulus),
            ("u", modulus - 1),
            ("s", modulus - 1),
        ):
            result = builder.call_program(
                controlled_bitwise_word_program(word, "xor_constant", pattern=constant),
                {"ctrl": terminals[0], "x": values[target]},
            )
            terminals[0], values[target] = result["ctrl"], result["x"]
    if adjoint:
        values["r"] = builder.call_program(addition, {"x": values["r"]})["x"]
        values["r"] = builder.call_program(complement, {"x": values["r"]})["x"]


def inverse_algorithm_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain the two 2n-bit logical histories across 2n shared iteration calls.

    Four words are MSB-first Montgomery values. Histories are opaque MSB words.
    Clean initialization and source-reachable or matching inverse inputs are
    required. The source all-ones modulus p+1 loader failure remains explicit.
    """
    word, bit = montgomery_register_type(width, modulus), quantum_bit_type()
    iteration = inverse_iteration_program(width, modulus, adjoint=adjoint)
    history = RegisterType("opaque", 2 * width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            u=word, v=word, r=word, s=word, m=history, f=bit, terminal_condition=history
        ),
        comparison_bit_functions(),
    )
    values = dict(builder.inputs)
    if not adjoint:
        _initialize(builder, values, word, modulus, adjoint)
    m, terminal = values.pop("m"), values.pop("terminal_condition")
    assert isinstance(m, RegisterBinding) and isinstance(terminal, RegisterBinding)
    markers, terminals = builder.split(m), builder.split(terminal)
    if adjoint:
        _correct(builder, values, terminals, word, modulus, adjoint)
    for i in reversed(range(2 * width)) if adjoint else range(2 * width):
        result = builder.call_program(
            iteration, {**values, "m": markers[i], "is_terminal": terminals[i]}
        )
        markers[i], terminals[i] = result.pop("m"), result.pop("is_terminal")
        values.update(result)
    if not adjoint:
        _correct(builder, values, terminals, word, modulus, adjoint)
    values["m"], values["terminal_condition"] = builder.join(markers), builder.join(
        terminals
    )
    if adjoint:
        _initialize(builder, values, word, modulus, adjoint)
    return builder.finalize(values)
