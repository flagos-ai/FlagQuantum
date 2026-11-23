# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _HalfLinearDepthGreaterThan and _HalfComparisonBase.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Own half-comparison difference/history outputs and consume their exact image."""

from typing import Literal

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.out_of_place_addition_component import out_of_place_addition_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from ._component_gates import comparison_bit_functions
from ._linear_words import (
    comparison_width,
    prepare_difference_words,
    restore_difference_words,
)


def half_greater_program(
    dtype: RegisterType, *, adjoint: bool = False
) -> ProgramRecord:
    """Create the widened difference and its sign flag, or erase both outputs.

    a/b are restored in both directions. c and target are owned RIGHT outputs
    forward and must contain the forward image when consumed as LEFT inputs.
    The inverse uses measurement-assisted sum erasure and clears temporary work.
    """
    width = comparison_width(dtype, allow_empty=True)
    if type(adjoint) is not bool:
        raise ValueError("Half comparison direction must be boolean")
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    word, bit = unsigned_register_type(width + 1), quantum_bit_type()
    signature = Signature(
        (
            Register("a", dtype),
            Register("b", dtype),
            Register("c", word, side=side),
            Register("target", bit, side=side),
        )
    )
    builder = ComponentProgramBuilder(signature, comparison_bit_functions())
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    a, b = prepare_difference_words(builder, dtype, a, b)
    complement = bitwise_word_program(word, "bitwise_not")
    if adjoint:
        c, target = builder.inputs["c"], builder.inputs["target"]
    else:
        result = builder.call_program(
            out_of_place_addition_program(width + 1, include_msb=False),
            {"a": a, "b": b},
        )
        a, b = result["a"], result["b"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        c = builder.call_program(complement, {"x": result["c"]})["x"]
    assert isinstance(c, RegisterBinding)
    cs = builder.split(c)
    if not adjoint:
        target = builder.allocate()
    copied = builder.call_dict("comparison_cx", {"ctrl": cs[0], "target": target})
    cs[0], target = copied["ctrl"], copied["target"]
    c = builder.join(cs)
    if adjoint:
        c = builder.call_program(complement, {"x": c})["x"]
        result = builder.call_program(
            out_of_place_addition_program(width + 1, include_msb=False, adjoint=True),
            {"a": a, "b": b, "c": c},
        )
        a, b = result["a"], result["b"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    a, b = restore_difference_words(builder, dtype, a, b)
    if adjoint:
        assert isinstance(target, RegisterBinding)
        builder.free(target)
        return builder.finalize({"a": a, "b": b})
    return builder.finalize({"a": a, "b": b, "c": c, "target": target})


def half_comparison_program(
    dtype: RegisterType,
    relation: Literal["gt", "ge", "lt", "le"],
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain the source relation wrappers and inverse output alias order.

    The source ge/lt inverse returns the two physical input aliases swapped.
    Callers must follow returned a/b bindings when chaining those inverses.
    """
    comparison_width(dtype, allow_empty=True)
    if relation not in {"gt", "ge", "lt", "le"} or type(adjoint) is not bool:
        raise ValueError(
            "Half comparison requires a supported relation and boolean direction"
        )
    base = half_greater_program(dtype, adjoint=adjoint)
    registers = base.functions[0].registers
    assert registers is not None
    builder = ComponentProgramBuilder(Signature(registers), comparison_bit_functions())
    values = dict(builder.inputs)
    inclusive = relation in {"ge", "le"}
    swapped = relation in {"ge", "lt"}
    if adjoint and inclusive:
        values["target"] = builder.call_dict("comparison_x", {"q": values["target"]})[
            "q"
        ]
    if swapped:
        values["a"], values["b"] = values["b"], values["a"]
    result = builder.inline_calls(base, values)
    if not adjoint:
        if swapped:
            result["a"], result["b"] = result["b"], result["a"]
        if inclusive:
            result["target"] = builder.call_dict(
                "comparison_x", {"q": result["target"]}
            )["q"]
    return builder.finalize(result)
