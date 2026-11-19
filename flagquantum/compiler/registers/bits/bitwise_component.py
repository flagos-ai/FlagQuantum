# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Xor, XorK and BitwiseNot decompositions.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Store bitwise word decompositions as owned, executable component calls."""

from typing import Literal

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .bitwise import xor_constant_ir, xor_ir
from .on_each import on_each_word_program


def bitwise_word_program(
    dtype: RegisterType,
    operation: Literal["xor", "xor_constant", "bitwise_not"],
    *,
    pattern: int | None = None,
) -> ProgramRecord:
    """Construct through-word XOR, encoded constant XOR, or complement.

    Words retain the supplied concrete dtype and MSB-first bit order. Constant
    XOR takes an already encoded unsigned bit pattern, including for signed
    words. Dtype value encoding belongs to the caller. Complement retains one
    OnEach child. No quantum workspace or classical slots are allocated.
    Symbolic and empty physical words fail at this finite execution boundary.
    """
    width = executable_register_width(dtype)
    if width < 1:
        raise ValueError("Bitwise component requires a positive concrete width")
    if operation not in {"xor", "xor_constant", "bitwise_not"}:
        raise ValueError("Unknown bitwise component operation")
    if operation != "xor_constant" and pattern is not None:
        raise ValueError("Only constant XOR accepts an encoded pattern")
    bit = quantum_bit_type()
    if operation == "bitwise_not":
        leaf = bit_gate_function(
            "bitwise_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
        )
        repeated = on_each_word_program(width, leaf, target_dtype=dtype)
        builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
        result = builder.call_program(repeated, {"q": builder.inputs["x"]})
        return builder.finalize({"x": result["q"]})
    if operation == "xor_constant":
        if pattern is None:
            raise ValueError("Constant XOR requires an encoded pattern")
        gates = xor_constant_ir(tuple(range(width)), pattern)
        signature = Signature.build_from_dtypes(x=dtype)
        leaf = bit_gate_function(
            "bitwise_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
        )
    else:
        gates = xor_ir(tuple(range(width)), tuple(range(width, 2 * width)))
        signature = Signature.build_from_dtypes(x=dtype, y=dtype)
        leaf = bit_gate_function(
            "bitwise_cx",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        )
    builder = ComponentProgramBuilder(signature, (leaf,))
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    lhs = builder.split(x)
    if operation == "xor_constant":
        for gate in gates:
            (index,) = gate.wires
            lhs[index] = builder.call_dict(leaf.name, {"q": lhs[index]})["q"]
        return builder.finalize({"x": builder.join(lhs, dtype)})
    y = builder.inputs["y"]
    assert isinstance(y, RegisterBinding)
    rhs = builder.split(y)
    for gate in gates:
        index, target = gate.wires
        other = target - width
        result = builder.call_dict(
            leaf.name, {"ctrl": lhs[index], "target": rhs[other]}
        )
        lhs[index], rhs[other] = result["ctrl"], result["target"]
    return builder.finalize(
        {"x": builder.join(lhs, dtype), "y": builder.join(rhs, dtype)}
    )
