# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix count_operations_sampled and SimInitInstruction.

"""Resolve physical register initialization for packed sampling commands."""

from ..compiler.formats._register_initialization import parse_initializations
from ..compiler.formats.kmx_circuit import KmxCircuit
from ..simulation.physical_basis_shots import RegisterInitialization


def parse_register_initialization(
    circuit: KmxCircuit, text: str
) -> tuple[RegisterInitialization, ...]:
    """Resolve source integer initialization strings to physical members."""
    assignments = []
    for instruction in parse_initializations(text):
        register = instruction.register
        if register >= len(circuit.registers):
            raise ValueError(f"Initialization register r{register} is not declared")
        value = instruction.value.value
        randomize = instruction.randomize
        if randomize and not value and instruction.value.width:
            raise ValueError("Random initialization modulus is zero")
        assignments.append(
            RegisterInitialization(
                circuit.registers[register].members, value, randomize
            )
        )
    return tuple(assignments)
