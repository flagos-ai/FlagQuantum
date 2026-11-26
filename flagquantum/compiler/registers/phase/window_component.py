# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/qpe_window_state.py.

"""Build allocated rectangular phase windows without algorithm ownership."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..bits.on_each import on_each_word_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function


def rectangular_window_program(width: SymbolicInt) -> ProgramRecord:
    """Retain Allocate and repeated-H calls, including a symbolic declaration.

    The RIGHT-only fractional output transfers ownership to the caller. An
    unresolved repeated-H body stays signature-only and cannot execute until
    supplied with a concrete definition.
    """
    dtype = fixed_point_register_type(width, width)
    signature = Signature((Register("qpe_reg", dtype, side=RegisterSide.RIGHT),))
    if type(width) is int and width < 0:
        raise ValueError("Rectangular window requires a nonnegative width")
    h = bit_gate_function(
        "hadamard",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("h", (0,)),)),
    )
    each = (
        on_each_word_program(width, h)
        if type(width) is int
        else ProgramRecord(
            (
                FunctionRecord(
                    "each_h",
                    (),
                    registers=(Register("q", RegisterType("opaque", width)),),
                ),
            )
        )
    )
    builder = ComponentProgramBuilder(signature)
    output = builder.allocate(dtype=dtype)
    returned = builder.call_program(each, {"q": output}, entry=each.functions[0].name)[
        "q"
    ]
    return builder.finalize({"qpe_reg": returned})
