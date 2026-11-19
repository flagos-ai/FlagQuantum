# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the complete Qualtran v0.7.0 Controlled tutorial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/Controlled.ipynb

"""Build the control tutorial through shared typed component transformations."""

import numpy as np

from examples.algorithms.adjoint_programs import serial_atoms
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    ControlAdder,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def _controlled_atom(name: str) -> ControlAdder:
    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = builder.call_dict(name, {"ctrl": control, "q": values["q"]})
        return result["ctrl"], (result["q"],)

    return add


def _control_x_calls(
    record: ProgramRecord,
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...],
) -> ProgramRecord:
    """Supply exact CX leaves while retaining the distinct source atom names."""
    atoms = record.functions[1:]
    bit = quantum_bit_type()
    definitions = tuple(
        bit_gate_function(
            f"controlled_{atom.name}",
            (Register("ctrl", bit), Register("q", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        )
        for atom in atoms
    )
    return control_component_via_predicate(
        ProgramRecord((*record.functions, *definitions)),
        {
            atom.name: _controlled_atom(leaf.name)
            for atom, leaf in zip(atoms, definitions, strict=True)
        },
        controls,
        patterns,
    )


def controlled_x_program(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...],
    *,
    atom_name: str = "x",
) -> ProgramRecord:
    """Apply X once under row-major word-equality controls and erase private work."""
    bit = quantum_bit_type()
    atom = bit_gate_function(
        atom_name, (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(Signature.build(q=1), (atom,))
    result = builder.call_dict(atom_name, builder.inputs)
    return _control_x_calls(builder.finalize(result), controls, patterns)


def controlled_serial_program() -> ProgramRecord:
    """Control the original three distinct serial X atoms with one positive bit."""
    return _control_x_calls(
        serial_atoms(),
        (Register("ctrl", quantum_bit_type()),),
        (ControlRegister(1, (1,)),),
    )


def controlled_each_x_program() -> ProgramRecord:
    """Share one two-bit predicate across three target X operations."""
    bit = quantum_bit_type()
    atom = bit_gate_function(
        "x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(Signature((Register("q", bit, (3,)),)), (atom,))
    outputs = [
        builder.call_dict("x", {"q": value})["q"]
        for value in np.asarray(builder.inputs["q"]).flat
    ]
    record = builder.finalize({"q": np.array(outputs, dtype=object)})
    return _control_x_calls(
        record, (Register("ctrl", bit, (2,)),), (ControlRegister(1, (1, 1)),)
    )
