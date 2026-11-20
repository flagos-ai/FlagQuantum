# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 multiplication.Square supplies the sided numeric contract.
# The frozen root supplies no executable decomposition.

"""Typed integer square with shared measured column-compression calls."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.weighted_columns_component import weighted_columns_xor_program
from ..component_builder import ComponentProgramBuilder
from .integer_square import _square_terms, integer_square_workspace_size


def integer_square_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Create a RIGHT square result or erase its matching LEFT image.

    Unsigned words are MSB first. Forward result storage starts clean. The
    inverse requires the coherent square image and releases its zeroed result.
    The column owner allocates width*(width-1) product/carry bits and restores
    them through shared And compute and measured erasure calls.
    """
    count = integer_square_workspace_size(width)
    if type(adjoint) is not bool:
        raise ValueError("Integer square direction requires a boolean")
    source_dtype = RegisterType("uint", width)
    result_dtype = RegisterType("uint", 2 * width)
    signature = Signature(
        (
            Register("a", source_dtype),
            Register(
                "result",
                result_dtype,
                side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
            ),
        )
    )
    builder = ComponentProgramBuilder(signature)
    source = builder.inputs["a"]
    assert isinstance(source, RegisterBinding)
    target = (
        builder.inputs["result"] if adjoint else builder.allocate(dtype=result_dtype)
    )
    assert isinstance(target, RegisterBinding)
    columns = weighted_columns_xor_program(
        width,
        2 * width,
        count,
        tuple(
            (column, width - 1 - i, width - 1 - j if i != j else None)
            for i, j, column in _square_terms(width)
        ),
    )
    result = builder.call_program(
        columns, {"input": builder.split(source), "result": builder.split(target)}
    )
    output = builder.join(np.asarray(result["result"]), result_dtype)
    restored = builder.join(np.asarray(result["input"]), source_dtype)
    if adjoint:
        builder.free(output)
        return builder.finalize({"a": restored})
    return builder.finalize({"a": restored, "result": output})
