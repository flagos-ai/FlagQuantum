# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared compression order derives from Qualtran v0.7.0 HammingWeightCompute.

"""Typed weighted-column XOR using owned product and carry lifetimes."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .bit_compression import weighted_column_components
from .bit_compression_structure import CompressionComponent
from .column_steps_component import append_column_steps, column_bit_functions


def weighted_columns_xor_program(
    input_width: int,
    result_width: int,
    workspace_width: int,
    terms: tuple[tuple[int, int, int | None], ...],
) -> ProgramRecord:
    """XOR the weighted polynomial into through bits and restore all input bits.

    Terms use input indices and little-endian column positions. Both ports are
    MSB-first bit arrays. The column graph owns clean workspace across shared And
    compute/erase calls. The caller supplies capacity for the exact column network.
    Repeating the program erases the mask from any occupied result.
    """
    if (
        any(type(w) is not int or w < 1 for w in (input_width, result_width))
        or type(workspace_width) is not int
        or workspace_width < 0
    ):
        raise ValueError("Weighted columns require positive ports and nonnegative work")
    for column, left, right in terms:
        if (
            type(column) is not int
            or not 0 <= column < result_width
            or type(left) is not int
            or not 0 <= left < input_width
            or right is not None
            and (
                type(right) is not int or not 0 <= right < input_width or right == left
            )
        ):
            raise ValueError(
                "Column terms require valid columns and distinct input bits"
            )
    work = tuple(range(input_width, input_width + workspace_width))
    try:
        compute, word, erase = weighted_column_components(result_width, work, terms)
    except (IndexError, StopIteration) as error:
        raise ValueError(
            "Column network exceeds result or workspace capacity"
        ) from error
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("input", bit, (input_width,)),
                Register("result", bit, (result_width,)),
            )
        ),
        column_bit_functions(),
    )
    wires: dict[int, RegisterBinding] = dict(
        enumerate(np.asarray(builder.inputs["input"]).flat)
    )
    if workspace_width:
        workspace = builder.allocate(workspace_width)
        wires.update(
            (input_width + i, value) for i, value in enumerate(builder.split(workspace))
        )
    copies: tuple[CompressionComponent, ...] = tuple(
        ("cx", (source, input_width + workspace_width + i))
        for i, source in reversed(tuple(enumerate(word)))
        if source is not None
    )
    wires.update(
        (input_width + workspace_width + i, value)
        for i, value in enumerate(np.asarray(builder.inputs["result"]))
    )
    for steps, inverse in ((compute, False), (copies, False), (erase, True)):
        append_column_steps(builder, wires, steps, inverse=inverse)
    if workspace_width:
        builder.free(
            builder.join(
                [wires[i] for i in work], RegisterType("opaque", workspace_width)
            )
        )
    return builder.finalize(
        {
            "input": np.array([wires[i] for i in range(input_width)], dtype=object),
            "result": np.array(
                [wires[input_width + workspace_width + i] for i in range(result_width)],
                dtype=object,
            ),
        }
    )
