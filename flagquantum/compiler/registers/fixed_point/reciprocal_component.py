# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Qualtran v0.7.0 InvertRealNumber supplies the mixed THRU signature and model.
# Native numerical construction follows reciprocal.py's widened Newton estimate.

"""Mixed fixed-point reciprocal masks over owned seed and Newton histories."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .reciprocal_newton_component import reciprocal_newton_program
from .reciprocal_seed_component import reciprocal_seed_program


def reciprocal_program(input_width: int, fraction: int) -> ProgramRecord:
    """XOR a lower reciprocal approximation into the original mixed THRU result.

    Source a and result share unsigned QFxp(input_width,fraction) storage. For
    a>=1, ceil(log2(input_width)) widened Newton iterations use input_width
    precision bits. Copy the leading fraction bits into the result's fractional
    suffix. Its integer prefix is preserved, including on occupied results.
    Below one, the image is zero. The finite lower estimate remains below one
    at a=1, so nonpositive fractional widths have a zero quantized image. Source
    and all owned histories are restored. Reapplying the XOR clears its image.
    """
    if (
        type(input_width) is not int
        or input_width < 1
        or type(fraction) is not int
        or fraction >= input_width
    ):
        raise ValueError(
            "Reciprocal requires positive concrete width and a nonempty integer part"
        )
    dtype = fixed_point_register_type(input_width, fraction)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=dtype, result=dtype)
    )
    source, result = builder.inputs["a"], builder.inputs["result"]
    assert isinstance(source, RegisterBinding) and isinstance(result, RegisterBinding)
    if fraction <= 0:
        return builder.finalize({"a": source, "result": result})
    seeded = builder.call_program(
        reciprocal_seed_program(input_width, fraction, input_width), {"a": source}
    )
    source, seed = seeded["a"], seeded["result"]
    assert isinstance(source, RegisterBinding) and isinstance(seed, RegisterBinding)
    history = [seed]
    pure = fixed_point_register_type(input_width, input_width)
    step = reciprocal_newton_program(input_width, fraction, input_width)
    for _ in range((input_width - 1).bit_length()):
        updated = builder.call_program(
            step,
            {
                "a": source,
                "estimate": history[-1],
                "result": builder.allocate(dtype=pure),
            },
        )
        source, previous, following = (
            updated["a"],
            updated["estimate"],
            updated["result"],
        )
        assert (
            isinstance(source, RegisterBinding)
            and isinstance(previous, RegisterBinding)
            and isinstance(following, RegisterBinding)
        )
        history[-1] = previous
        history.append(following)
    latest = builder.split(history[-1])
    output = builder.split(result)
    small = RegisterType("uint", fraction)
    copied = builder.call_program(
        bitwise_word_program(small, "xor"),
        {
            "x": builder.join(latest[:fraction], small),
            "y": builder.join(output[-fraction:], small),
        },
    )
    estimate, suffix = copied["x"], copied["y"]
    assert isinstance(estimate, RegisterBinding) and isinstance(suffix, RegisterBinding)
    history[-1] = builder.join([*builder.split(estimate), *latest[fraction:]], pure)
    result = builder.join([*output[:-fraction], *builder.split(suffix)], dtype)
    for index in reversed(range(len(history) - 1)):
        erased = builder.call_program(
            step,
            {"a": source, "estimate": history[index], "result": history[index + 1]},
        )
        source, previous, cleared = erased["a"], erased["estimate"], erased["result"]
        assert (
            isinstance(source, RegisterBinding)
            and isinstance(previous, RegisterBinding)
            and isinstance(cleared, RegisterBinding)
        )
        history[index] = previous
        builder.free(cleared)
    restored = builder.call_program(
        reciprocal_seed_program(input_width, fraction, input_width, adjoint=True),
        {"a": source, "result": history[0]},
    )
    return builder.finalize({"a": restored["a"], "result": result})
