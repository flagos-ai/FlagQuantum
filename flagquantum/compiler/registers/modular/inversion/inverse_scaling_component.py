# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep6.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose controlled cyclic division, modular doubling and parity cleanup."""

from .....core.ir import CircuitIR
from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_bindings import RegisterBinding
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.registers import Register, RegisterType
from .....core.ir.program.signatures import Signature
from ...comparison._component_gates import comparison_bit_functions
from ...component_builder import BindingCollection, ComponentProgramBuilder
from ...component_gate_function import bit_gate_function
from ...primitives.controlled_swap import controlled_swap_bit_ir
from ..modular_doubling_component import modular_doubling_program
from ..modular_halving_component import modular_halving_program
from .inverse_exchange_component import inverse_exchange_program


def inverse_scaling_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve logical words/flags while following the changed r storage alias.

    Words are MSB-first. Active v must be even for the cyclic division to agree
    with the source callback. r must be a residue, and inverse inputs must be
    a forward image. Temporary work returns clean on that domain.
    """
    word, bit = montgomery_register_type(width, modulus), quantum_bit_type()
    if type(adjoint) is not bool:
        raise ValueError("Inverse scaling direction requires a boolean")
    scaling = (
        modular_halving_program(word, modulus)
        if adjoint
        else modular_doubling_program(word, modulus)
    )
    rotation = bit_gate_function(
        "inverse_rotate_bit",
        (Register("ctrl", bit), Register("x", bit), Register("y", bit)),
        CircuitIR(3, controlled_swap_bit_ir(0, 1, 2)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            u=word, v=word, r=word, s=word, b=bit, a=bit, m=bit, f=bit
        ),
        (*comparison_bit_functions(), rotation),
    )
    values = dict(builder.inputs)
    if adjoint:
        _update_parity(builder, values, word)
        values.update(
            builder.inline_calls(inverse_exchange_program(width, adjoint=True), values)
        )
    else:
        _update_branch(builder, values, adjoint=False)
        _rotate(builder, values, word, adjoint=False)
    values["r"] = builder.call_program(scaling, {"x": values["r"]})["x"]
    if adjoint:
        _rotate(builder, values, word, adjoint=True)
        _update_branch(builder, values, adjoint=True)
    else:
        values.update(builder.inline_calls(inverse_exchange_program(width), values))
        _update_parity(builder, values, word)
    return builder.finalize(values)


def _update_branch(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    *,
    adjoint: bool,
) -> None:
    for control in (("a", "m") if adjoint else ("m", "a")):
        copied = builder.call_dict(
            "comparison_cx", {"ctrl": values[control], "target": values["b"]}
        )
        values[control], values["b"] = copied["ctrl"], copied["target"]


def _rotate(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    word: RegisterType,
    *,
    adjoint: bool,
) -> None:
    v = values["v"]
    assert isinstance(v, RegisterBinding)
    pieces = builder.split(v)
    count = len(pieces) - 1
    for i in range(count) if adjoint else reversed(range(count)):
        swapped = builder.call_dict(
            "inverse_rotate_bit",
            {"ctrl": values["f"], "x": pieces[i], "y": pieces[i + 1]},
        )
        values["f"], pieces[i], pieces[i + 1] = (
            swapped["ctrl"],
            swapped["x"],
            swapped["y"],
        )
    values["v"] = builder.join(pieces, word)


def _update_parity(
    builder: ComponentProgramBuilder,
    values: dict[str, BindingCollection],
    word: RegisterType,
) -> None:
    s = values["s"]
    assert isinstance(s, RegisterBinding)
    pieces = builder.split(s)
    pieces[-1] = builder.call_dict("comparison_x", {"q": pieces[-1]})["q"]
    copied = builder.call_dict(
        "comparison_cx", {"ctrl": pieces[-1], "target": values["a"]}
    )
    pieces[-1], values["a"] = copied["ctrl"], copied["target"]
    pieces[-1] = builder.call_dict("comparison_x", {"q": pieces[-1]})["q"]
    values["s"] = builder.join(pieces, word)
