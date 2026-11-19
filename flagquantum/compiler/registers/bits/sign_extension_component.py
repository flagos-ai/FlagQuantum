# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SignExtend and SignTruncate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/sign_extension.py

"""Own the redundant sign prefix through extension and exact truncation."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from .fanout_component import fanout_component_program


def sign_extension_program(
    input_dtype: RegisterType, output_dtype: RegisterType, *, adjoint: bool = False
) -> ProgramRecord:
    """Consume a signed word and return its wider image, or consume that image.

    Both arguments describe the forward extension. Bits are MSB first and use
    the same signed encoding. The adjoint clears the correct redundant prefix,
    frees it and restores the narrow word. An arbitrary prefix cannot be freed.
    """
    source_width = executable_register_width(input_dtype)
    target_width = executable_register_width(output_dtype)
    if (
        input_dtype.encoding not in {"signed", "ones_complement"}
        or input_dtype.encoding != output_dtype.encoding
        or source_width < 1
        or target_width <= source_width
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Sign extension requires increasing matching signed widths and a boolean direction"
        )
    count = target_width - source_width
    before, after = (
        (output_dtype, input_dtype) if adjoint else (input_dtype, output_dtype)
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", before, side=RegisterSide.LEFT),
                Register("y", after, side=RegisterSide.RIGHT),
            )
        )
    )
    word = builder.inputs["x"]
    assert isinstance(word, RegisterBinding)
    targets: BindingCollection
    if adjoint:
        bits = builder.split(word)
        prefix, bits = bits[:count], bits[count:]
        sign = bits[0]
        targets = builder.join(prefix)
    else:
        targets = builder.allocate(count)
        bits = builder.split(word)
        sign = bits[0]
    result = builder.call_program(
        fanout_component_program(count), {"control": sign, "targets": targets}
    )
    bits[0] = result["control"]
    targets = result["targets"]
    assert isinstance(targets, RegisterBinding)
    if adjoint:
        builder.free(targets)
        # The source truncation's untyped Join remains a zero-cost output view.
        output = builder.join(bits)
    else:
        output = builder.join([*builder.split(targets), *bits], output_dtype)
    return builder.finalize({"y": output})
