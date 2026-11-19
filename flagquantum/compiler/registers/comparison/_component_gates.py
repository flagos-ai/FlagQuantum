"""Shared one-bit leaves for the comparison component owners."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_gate_function import bit_gate_function


def comparison_bit_functions() -> tuple[FunctionRecord, FunctionRecord]:
    """Return X and CX with the source's named QBit ports."""
    bit = quantum_bit_type()
    return (
        bit_gate_function(
            "comparison_x",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        ),
        bit_gate_function(
            "comparison_cx",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        ),
    )
