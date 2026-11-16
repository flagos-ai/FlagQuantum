# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/qubit_count_many_alloc.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/qubit_count_many_alloc.py

"""Three source scheduling fixtures with the same phase action."""

from typing import Literal

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

AllocationMode = Literal["once", "many", "abstracted"]


def _phase_step(
    builder: ComponentProgramBuilder, control: RegisterBinding, ancilla: RegisterBinding
) -> tuple[RegisterBinding, RegisterBinding]:
    first = builder.call_dict("allocation_cx", {"ctrl": control, "target": ancilla})
    phase = builder.call_dict("allocation_t", {"q": first["target"]})["q"]
    last = builder.call_dict("allocation_cx", {"ctrl": first["ctrl"], "target": phase})
    acted, restored = last["ctrl"], last["target"]
    assert isinstance(acted, RegisterBinding) and isinstance(restored, RegisterBinding)
    return acted, restored


def many_allocation_fixture(width: int, mode: AllocationMode) -> ProgramRecord:
    """Declare the original allocation scopes and ordered CNOT/T/CNOT calls."""
    if width < 1:
        raise ValueError("Allocation fixture requires a positive register width")
    bit = quantum_bit_type()
    primitives = (
        bit_gate_function(
            "allocation_cx",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        ),
        bit_gate_function(
            "allocation_t",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("t", (0,)),)),
        ),
    )
    definitions: tuple[FunctionRecord, ...] = primitives
    if mode == "abstracted":
        inner = ComponentProgramBuilder(Signature.build(x=1), primitives, name="inner")
        x = inner.inputs["x"]
        assert isinstance(x, RegisterBinding)
        x, scratch = _phase_step(inner, x, inner.allocate(dtype=bit))
        inner.free(scratch)
        definitions = inner.finalize({"x": x}).functions

    signature = Signature.build(x=width)
    builder = ComponentProgramBuilder(signature, definitions)
    word = builder.inputs["x"]
    assert isinstance(word, RegisterBinding)
    bits = builder.split(word)
    if mode == "once":
        scratch = builder.allocate(dtype=bit)
        for index, payload in enumerate(bits):
            bits[index], scratch = _phase_step(builder, payload, scratch)
        builder.free(scratch)
    elif mode == "many":
        for index, payload in enumerate(bits):
            bits[index], scratch = _phase_step(
                builder, payload, builder.allocate(dtype=bit)
            )
            builder.free(scratch)
    else:
        for index, payload in enumerate(bits):
            bits[index] = builder.call_dict("inner", {"x": payload})["x"]
    return builder.finalize({"x": builder.join(bits, word.register.dtype)})
