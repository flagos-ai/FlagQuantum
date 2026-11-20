# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared compression order derives from Qualtran v0.7.0 HammingWeightCompute.

"""Typed weighted-column XOR using owned product and carry lifetimes."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .bit_compression import weighted_column_components


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
    cx = bit_gate_function(
        "column_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    conjunction = bit_gate_function(
        "column_and",
        (Register("bits", bit, (3,)),),
        CircuitIR(3, and_compute_ir(0, 1, 2)),
    )
    erasure = bit_gate_function(
        "column_erase",
        (Register("bits", bit, (3,)),),
        CircuitIR(3, and_uncompute_ir(0, 1, 2, 0)),
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("input", bit, (input_width,)),
                Register("result", bit, (result_width,)),
            )
        ),
        (cx, conjunction, erasure),
    )
    wires: dict[int, RegisterBinding] = dict(
        enumerate(np.asarray(builder.inputs["input"]).flat)
    )
    if workspace_width:
        workspace = builder.allocate(workspace_width)
        wires.update(
            (input_width + i, value) for i, value in enumerate(builder.split(workspace))
        )
    copies = tuple(
        ("cx", (source, input_width + workspace_width + i))
        for i, source in reversed(tuple(enumerate(word)))
        if source is not None
    )
    wires.update(
        (input_width + workspace_width + i, value)
        for i, value in enumerate(np.asarray(builder.inputs["result"]))
    )
    for steps, inverse in ((compute, False), (copies, False), (erase, True)):
        for kind, addresses in steps:
            x, y = addresses[:2]
            if kind == "cx":
                result = builder.call_dict(
                    "column_cx", {"ctrl": wires[x], "target": wires[y]}
                )
                control, target = result["ctrl"], result["target"]
                assert isinstance(control, RegisterBinding) and isinstance(
                    target, RegisterBinding
                )
                wires[x], wires[y] = control, target
            else:
                result = builder.call_dict(
                    "column_erase" if inverse else "column_and",
                    {"bits": np.array([wires[i] for i in addresses], dtype=object)},
                )
                returned = np.asarray(result["bits"])
                wires.update(zip(addresses, returned, strict=True))
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
