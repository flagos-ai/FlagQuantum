# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix count_operations_sampled and SimInitInstruction.

"""Resolve physical register initialization for packed sampling commands."""

import re

from ..compiler.formats.kmx_circuit import KmxCircuit
from ..compiler.formats.unsigned_word import parse_unsigned_word
from ..simulation.physical_basis_shots import RegisterInitialization

_INIT = re.compile(r"r([0-9]*)([=<])(.*)")


def parse_register_initialization(
    circuit: KmxCircuit, text: str
) -> tuple[RegisterInitialization, ...]:
    """Resolve source integer initialization strings to physical members."""
    assignments = []
    for item in text.removesuffix(",").split(",") if text else ():
        match = _INIT.fullmatch(item)
        if match is None:
            raise ValueError(f"Invalid register initialization {item!r}")
        index, operator, payload = match.groups()
        register = int(index or "0")
        if register >= len(circuit.registers):
            raise ValueError(f"Initialization register r{register} is not declared")
        randomize = operator == "<" or payload == "*"
        if operator == "=" and payload == "*":
            value = 0
        else:
            value, width = parse_unsigned_word(payload)
            if randomize and not value and width:
                raise ValueError("Random initialization modulus is zero")
        assignments.append(
            RegisterInitialization(
                circuit.registers[register].members, value, randomize
            )
        )
    return tuple(assignments)
