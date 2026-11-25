# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 programmable_ancilla_rotation.py.

"""Bind the locked preparation and programmed rotation to a native circuit."""

from collections import Counter
from dataclasses import dataclass, field
from math import pi

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.phase.programmed_phase import (
    programmed_phase_ir,
    programmed_phase_rotation_ir,
)
from flagquantum.compiler.registers.phase.programmed_phase_resource_model import (
    ProgrammedAncillaCall,
    ProgrammedRotationModel,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class ProgrammedAncillaExample:
    model: ProgrammedAncillaCall = field(
        default_factory=lambda: ProgrammedAncillaCall(pi / 4)
    )

    @property
    def signature(self) -> Signature:
        return Signature((Register("q", quantum_bit_type(), side=RegisterSide.RIGHT),))

    def append_to(self, circuit: Circuit) -> Circuit:
        theta, epsilon = self.model.exponent, self.model.precision
        if not isinstance(theta, (int, float)) or not isinstance(epsilon, (int, float)):
            raise ValueError("Bind the programmed state parameters before appending")
        gates = programmed_phase_ir(0, exponent=theta, epsilon=epsilon)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


@dataclass(frozen=True)
class ProgrammedRotationExample:
    model: ProgrammedRotationModel = field(
        default_factory=lambda: ProgrammedRotationModel(pi / 4)
    )

    @property
    def signature(self) -> Signature:
        return Signature((Register("q", quantum_bit_type()),))

    def append_to(self, circuit: Circuit) -> Circuit:
        model = self.model
        theta, epsilon, rounds = model.exponent, model.precision, model.rounds
        if (
            not isinstance(theta, (int, float))
            or not isinstance(epsilon, (int, float))
            or not isinstance(rounds, int)
        ):
            raise ValueError("Bind the programmed rotation parameters before appending")
        if rounds:
            gates = programmed_phase_rotation_ir(
                0,
                1,
                tuple(range(rounds)),
                exponent=theta,
                epsilon=epsilon,
                apply_final_correction=model.apply_final_correction,
                early_exit=True,
            )
        elif model.apply_final_correction:
            # With no attempts the terminal correction is unconditional.
            _, rotation = programmed_phase_ir(0, exponent=theta, epsilon=epsilon)
            gates = (rotation,)
        else:
            gates = ()
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    op = ProgrammedRotationExample()
    circuit = DynamicCircuit(2)
    circuit.append(op)
    counts = Counter(gate.name for gate in circuit.to_ir().instructions)
    assert counts == {
        **dict.fromkeys(("h", "phase", "cx", "measure", "x"), 2),
        "classical_xor": 1,
    }
    print("Emitted instructions", dict(counts))
    print("Source worst-case logical resources", op.model.resource_formula())
    print("Source precision-sensitive T estimate", op.model.synthesis_t_count())


if __name__ == "__main__":
    main()
