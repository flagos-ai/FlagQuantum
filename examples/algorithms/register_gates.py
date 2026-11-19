# Copyright 2023 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from the complete Qualtran v0.7.0 GateWithRegisters tutorial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/gate_with_registers.ipynb

"""Bind a named-register operation through shared Program calls and append."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class RegisterControlledTargets:
    """Apply CX for each ordered control/target pair using named register ports."""

    control: tuple[int, ...]
    target: tuple[int, ...]

    @property
    def signature(self) -> Signature:
        return Signature.build(control=len(self.control), target=len(self.target))

    def program(self) -> ProgramRecord:
        """Store the original nested loop as shared typed CX calls."""
        if not self.control or not self.target:
            raise ValueError(
                "The register example requires nonempty control and target"
            )
        bit = quantum_bit_type()
        cx = bit_gate_function(
            "cx",
            (Register("control", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        )
        builder = ComponentProgramBuilder(self.signature, (cx,))
        control_word, target_word = builder.inputs["control"], builder.inputs["target"]
        assert isinstance(control_word, RegisterBinding)
        assert isinstance(target_word, RegisterBinding)
        controls, targets = builder.split(control_word), builder.split(target_word)
        for i in range(len(controls)):
            for j in range(len(targets)):
                result = builder.call_dict(
                    "cx", {"control": controls[i], "target": targets[j]}
                )
                controls[i], targets[j] = result["control"], result["target"]
        return builder.finalize(
            {
                "control": builder.join(controls, control_word.register.dtype),
                "target": builder.join(targets, target_word.register.dtype),
            }
        )

    def bind(self) -> BoundRegisterProgram:
        return bind_register_program(self.program(), (self.control, self.target))

    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.compose(self.bind().circuit)
