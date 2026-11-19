# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Subtract and SubtractFrom.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py

"""Compose two subtraction directions with owned signed prefixes and typed Add."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..bits.fanout_component import fanout_component_program
from ..bits.on_each import on_each_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..layout.component_cast import cast_word_function
from .addition_component import addition_component_program


def subtraction_program(
    a_dtype: RegisterType, b_dtype: RegisterType | None = None
) -> ProgramRecord:
    """Preserve a and write a-b into b with the original MSB-first calls.

    Concrete integer descriptors require 0 < width(a) <= width(b). Signed a
    sign-extends into an owned prefix, other encodings zero-extend. The prefix
    and shared Add carries return clean. Invalid descriptors raise ValueError.
    """
    target_dtype = a_dtype if b_dtype is None else b_dtype
    left, width = executable_register_width(a_dtype), executable_register_width(
        target_dtype
    )
    if not 1 <= left <= width or any(
        dtype.encoding not in {"uint", "signed", "montgomery"}
        for dtype in (a_dtype, target_dtype)
    ):
        raise ValueError(
            "Subtract requires concrete integer descriptors with left <= right"
        )
    delta = width - left
    unsigned, small = RegisterType("uint", width), RegisterType("uint", left)
    prefix_type = RegisterType("opaque", delta)
    addition = addition_component_program(width, name="subtract_add")
    x = bit_gate_function(
        "subtract_x",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
    )
    complement = on_each_word_program(width, x)
    casts = (
        (
            cast_word_function("subtract_cast_a", a_dtype, unsigned),
            cast_word_function("subtract_cast_b", target_dtype, unsigned),
            cast_word_function("subtract_restore_b", unsigned, target_dtype),
            cast_word_function("subtract_restore_a", small, a_dtype),
        )
        if not delta
        else (
            cast_word_function("subtract_cast_b", target_dtype, unsigned),
            cast_word_function("subtract_restore_b", unsigned, target_dtype),
            cast_word_function("subtract_restore_a", small, a_dtype),
            cast_word_function(
                "subtract_cast_prefix", RegisterType("opaque", delta), prefix_type
            ),
        )
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=a_dtype, b=target_dtype),
        (*casts, *addition.functions),
    )
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding)
    if delta:
        bits = builder.split(a)
        prefix: BindingCollection = builder.allocate(dtype=prefix_type)
        if a_dtype.encoding == "signed":
            extended = builder.call_program(
                fanout_component_program(delta), {"control": bits[0], "targets": prefix}
            )
            bits[0], prefix = extended["control"], extended["targets"]
        assert isinstance(prefix, RegisterBinding)
        a = builder.join([*builder.split(prefix), *bits], unsigned)
    else:
        a = builder.call_dict("subtract_cast_a", {"reg": a})["reg"]
    b = builder.call_dict("subtract_cast_b", {"reg": b})["reg"]
    a = builder.call_program(complement, {"q": a})["q"]
    result = builder.call_dict("subtract_add", {"a": a, "b": b})
    a, b = result["a"], result["b"]
    b = builder.call_program(complement, {"q": b})["q"]
    a = builder.call_program(complement, {"q": a})["q"]
    b = builder.call_dict("subtract_restore_b", {"reg": b})["reg"]
    if delta:
        assert isinstance(a, RegisterBinding)
        bits = builder.split(a)
        prefix = builder.join(bits[:delta])
        if a_dtype.encoding == "signed":
            extended = builder.call_program(
                fanout_component_program(delta),
                {"control": bits[delta], "targets": prefix},
            )
            bits[delta], prefix = extended["control"], extended["targets"]
        prefix = builder.call_dict("subtract_cast_prefix", {"reg": prefix})["reg"]
        assert isinstance(prefix, RegisterBinding)
        builder.free(prefix)
        a = builder.join(bits[delta:], small)
    a = builder.call_dict("subtract_restore_a", {"reg": a})["reg"]
    return builder.finalize({"a": a, "b": b})


def subtract_from_program(dtype: RegisterType) -> ProgramRecord:
    """Preserve a and write b-a into equal-width typed b, MSB first.

    Shared Add owns measured clean carries. Both input words remain live.
    Invalid or nonconcrete integer descriptors raise ValueError through Add.
    """
    width = executable_register_width(dtype)
    addition = addition_component_program(width, left_dtype=dtype, right_dtype=dtype)
    complement = bitwise_word_program(dtype, "bitwise_not")
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(a=dtype, b=dtype))
    a, b = builder.inputs["a"], builder.inputs["b"]
    b = builder.call_program(complement, {"x": b})["x"]
    result = builder.call_program(addition, {"a": a, "b": b})
    a, b = result["a"], result["b"]
    b = builder.call_program(complement, {"x": b})["x"]
    return builder.finalize({"a": a, "b": b})
