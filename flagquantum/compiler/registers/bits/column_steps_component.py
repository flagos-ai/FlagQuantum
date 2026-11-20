# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared typed steps follow the existing Qualtran HammingWeightCompute-derived
# column compression sequence in bit_compression.py.

"""Shared typed CNOT and measured And calls for live column consumers."""

from collections.abc import Iterable

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .bit_compression_structure import CompressionComponent


def column_bit_functions() -> tuple[FunctionRecord, ...]:
    """Return the existing column CX, conjunction and immediate-feedback leaves."""
    bit = quantum_bit_type()
    return (
        bit_gate_function(
            "column_cx",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        ),
        bit_gate_function(
            "column_and",
            (Register("bits", bit, (3,)),),
            CircuitIR(3, and_compute_ir(0, 1, 2)),
        ),
        bit_gate_function(
            "column_erase",
            (Register("bits", bit, (3,)),),
            CircuitIR(3, and_uncompute_ir(0, 1, 2, 0)),
        ),
    )


def append_column_steps(
    builder: ComponentProgramBuilder,
    wires: dict[int, RegisterBinding],
    steps: Iterable[CompressionComponent],
    *,
    inverse: bool = False,
) -> None:
    """Append the shared sequence and retain every returned SSA wire binding."""
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
            wires.update(zip(addresses, np.asarray(result["bits"]), strict=True))
