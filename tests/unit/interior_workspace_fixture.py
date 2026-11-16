# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/interior_alloc.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/interior_alloc.py

"""Construct the source allocate/swap/release sequence in typed register IR."""

from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.primitives.swap import swap_bit_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def interior_workspace_signature(width: SymbolicInt) -> Signature:
    return Signature.build(x=width, y=width)


def _word_swap(width: int) -> ProgramRecord:
    bit = quantum_bit_type()
    dtype = opaque_register_type(width)
    swap = bit_gate_function(
        "swap_bit",
        (Register("x", bit), Register("y", bit)),
        CircuitIR(2, swap_bit_ir(0, 1)),
    )
    builder = ComponentProgramBuilder(
        Signature((Register("x", dtype), Register("y", dtype))),
        (swap,),
        name="swap_word",
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    pairs = [
        builder.call_dict("swap_bit", {"x": left, "y": right})
        for left, right in zip(builder.split(x), builder.split(y), strict=True)
    ]
    return builder.finalize(
        {
            name: builder.join([pair[name] for pair in pairs], dtype)
            for name in ("x", "y")
        }
    )


def interior_workspace_fixture(width: int) -> ProgramRecord:
    """Route x into y on the clean-y input subspace; the middle word is freed."""
    dtype = opaque_register_type(width)
    signature = Signature((Register("x", dtype), Register("y", dtype)))
    builder = ComponentProgramBuilder(signature, _word_swap(width).functions)
    middle = builder.allocate(dtype=dtype)
    first = builder.call_dict("swap_word", {"x": builder.inputs["x"], "y": middle})
    second = builder.call_dict("swap_word", {"x": first["y"], "y": builder.inputs["y"]})
    assert isinstance(second["x"], RegisterBinding)
    builder.free(second["x"])
    return builder.finalize({"x": first["x"], "y": second["y"]})


def interior_call_kinds(record: ProgramRecord) -> tuple[str, ...]:
    functions: dict[str, FunctionRecord] = {fn.name: fn for fn in record.functions}
    names = []
    for call in functions["main"].blocks[0].operations:
        assert call.callee is not None
        callee = functions[call.callee]
        opcode = callee.blocks[0].operations[0].opcode
        names.append(
            "Swap"
            if callee.name == "swap_word"
            else {"allocate_register": "Allocate", "free_register": "Free"}[opcode]
        )
    return tuple(names)
