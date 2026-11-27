"""Shared typed CX and Toffoli calls for recursive field product components."""

import numpy as np
from numpy.typing import NDArray

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function


def product_bit_functions() -> tuple[FunctionRecord, FunctionRecord]:
    bit = quantum_bit_type()
    return (
        bit_gate_function(
            "product_cx",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        ),
        bit_gate_function(
            "product_ccx",
            (Register("ctrl", bit, (2,)), Register("target", bit)),
            CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
        ),
    )


def product_cx(
    builder: ComponentProgramBuilder, control: object, target: object
) -> tuple[BindingCollection, BindingCollection]:
    result = builder.call_dict("product_cx", {"ctrl": control, "target": target})
    return result["ctrl"], result["target"]


def product_ccx(
    builder: ComponentProgramBuilder, left: object, right: object, target: object
) -> tuple[BindingCollection, BindingCollection, BindingCollection]:
    result = builder.call_dict(
        "product_ccx", {"ctrl": np.array([left, right], dtype=object), "target": target}
    )
    controls = np.asarray(result["ctrl"], dtype=object)
    return controls[0], controls[1], result["target"]


def field_word_call(
    builder: ComponentProgramBuilder,
    child: ProgramRecord,
    port: str,
    coefficients: NDArray[np.object_],
    dtype: RegisterType,
) -> NDArray[np.object_]:
    """Call a typed MSB word child from a coefficient-zero-first bit view."""
    word = builder.join(coefficients[::-1], dtype)
    result = builder.call_program(child, {port: word}, entry=child.functions[0].name)[
        port
    ]
    assert isinstance(result, RegisterBinding)
    return builder.split(result)[::-1]
