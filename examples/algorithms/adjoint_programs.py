# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Examples adapted from the complete Qualtran v0.7.0 Adjoint tutorial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/Adjoint.ipynb

"""Compose preparation, serial and phase examples using native register owners."""

import numpy as np

from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature


def tutorial_definitions() -> tuple[FunctionRecord, ...]:
    """Keep primitive bodies and one-sided state/effect ownership explicit."""
    bit = quantum_bit_type()
    primitives = tuple(
        bit_gate_function(
            name, (Register("q", bit),), CircuitIR(1, (Instruction(name, (0,)),))
        )
        for name in ("h", "t", "tdg", "x")
    )
    cx = bit_gate_function(
        "cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    zero = integer_basis_function("zero", bit, 0)
    effect = integer_basis_function("zero_effect", bit, 0, state=False)
    plus = ComponentProgramBuilder(
        Signature((Register("val", bit, side=RegisterSide.RIGHT),)),
        (zero, primitives[0]),
        name="plus",
    )
    value = plus.call_dict("zero", {})["val"]
    value = plus.call_dict("h", {"q": value})["q"]
    plus_definition = plus.finalize({"val": value}).functions[0]
    plus_effect = ComponentProgramBuilder(
        Signature((Register("val", bit, side=RegisterSide.LEFT),)),
        (effect, primitives[0]),
        name="plus_effect",
    )
    value = plus_effect.call_dict("h", {"q": plus_effect.inputs["val"]})["q"]
    plus_effect.call_dict("zero_effect", {"val": value})
    effect_definition = plus_effect.finalize({}).functions[0]
    return (*primitives, cx, zero, effect, plus_definition, effect_definition)


def bell_preparation() -> ProgramRecord:
    """Prepare the tutorial's four-bit GHZ chain with four owned outputs."""
    builder = ComponentProgramBuilder(
        Signature(()), tutorial_definitions(), allow_register_addition=True
    )
    current = builder.call_dict("plus", {})["val"]
    outputs = []
    for _ in range(3):
        next_bit = builder.call_dict("zero", {})["val"]
        result = builder.call_dict("cx", {"ctrl": current, "target": next_bit})
        outputs.append(result["ctrl"])
        current = result["target"]
    outputs.append(current)
    return builder.finalize({"qs": np.array(outputs, dtype=object)})


def serial_atoms() -> ProgramRecord:
    """Retain the three distinguishable X atoms in the source serial guide."""
    bit = quantum_bit_type()
    atoms = tuple(
        bit_gate_function(
            f"atom{index}",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        )
        for index in range(3)
    )
    builder = ComponentProgramBuilder(Signature((Register("reg", bit),)), atoms)
    value = builder.inputs["reg"]
    for atom in atoms:
        value = builder.call_dict(atom.name, {"q": value})["q"]
    return builder.finalize({"reg": value})


def phase_program() -> ProgramRecord:
    """Store H followed by T with the original tutorial's through port x."""
    builder = ComponentProgramBuilder(Signature.build(x=1), tutorial_definitions())
    value = builder.call_dict("h", {"q": builder.inputs["x"]})["q"]
    value = builder.call_dict("t", {"q": value})["q"]
    return builder.finalize({"x": value})


def tutorial_adjoint(record: ProgramRecord) -> ProgramRecord:
    """Use explicit exact leaves and derive every stored composite above them."""
    return adjoint_component_hierarchy(
        record,
        {
            "zero": "zero_effect",
            "plus": "plus_effect",
            "h": "h",
            "cx": "cx",
            "x": "x",
            "t": "tdg",
            "tdg": "t",
            **{f"atom{index}": f"atom{index}" for index in range(3)},
        },
    )
