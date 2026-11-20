# SPDX-License-Identifier: Apache-2.0

"""Shared typed result ownership for unsigned column arithmetic."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.weighted_columns_component import weighted_columns_xor_program
from ..component_builder import BindingCollection, ComponentProgramBuilder


def column_result_program(
    inputs: tuple[Register, ...],
    result_width: int,
    workspace_width: int,
    terms: tuple[tuple[int, int, int | None], ...],
    *,
    adjoint: bool,
    result_dtype: RegisterType | None = None,
) -> ProgramRecord:
    """Preserve source words and create or consume their polynomial image.

    Callers own finite unsigned input shapes, term generation and width checks.
    Terms index their concatenated MSB-first words in row-major port order.
    This owner handles result storage, views and the shared column call only.
    """
    if result_dtype is None:
        result_dtype = RegisterType("uint", result_width)
    builder = ComponentProgramBuilder(
        Signature(
            (
                *inputs,
                Register(
                    "result",
                    result_dtype,
                    side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
                ),
            )
        )
    )
    target = (
        builder.inputs["result"] if adjoint else builder.allocate(dtype=result_dtype)
    )
    assert isinstance(target, RegisterBinding)
    words: list[tuple[RegisterType, int]] = []
    bits: list[RegisterBinding] = []
    for register in inputs:
        for binding in np.asarray(builder.inputs[register.name]).flat:
            assert isinstance(binding, RegisterBinding)
            word_bits = builder.split(binding)
            words.append((binding.register.dtype, len(word_bits)))
            bits.extend(word_bits)
    result = builder.call_program(
        weighted_columns_xor_program(len(bits), result_width, workspace_width, terms),
        {"input": np.array(bits, dtype=object), "result": builder.split(target)},
    )
    output = builder.join(np.asarray(result["result"]), result_dtype)
    returned = np.asarray(result["input"])
    offset = 0
    restored: list[RegisterBinding] = []
    for dtype, width in words:
        restored.append(builder.join(returned[offset : offset + width], dtype))
        offset += width
    outputs: dict[str, BindingCollection] = {}
    position = 0
    for register in inputs:
        size = int(np.prod(register.shape))
        values = restored[position : position + size]
        outputs[register.name] = (
            np.array(values, dtype=object).reshape(register.shape)
            if register.shape
            else values[0]
        )
        position += size
    if adjoint:
        builder.free(output)
    else:
        outputs["result"] = output
    return builder.finalize(outputs)
