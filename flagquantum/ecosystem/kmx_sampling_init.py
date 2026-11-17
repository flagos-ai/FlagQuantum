# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix count_operations_sampled and SimInitInstruction.

"""Resolve physical register initialization for packed sampling commands."""

import re

from ..compiler.formats.kmx_circuit import KmxCircuit
from ..simulation.physical_basis_shots import RegisterInitialization

_INIT = re.compile(r"r([0-9]*)([=<])(.*)")
_VALUE = re.compile(r"0b[01_]*|0x[0-9a-fA-F_]*|[0-9_]+")


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
            payload = payload.strip(" ")
            if not _VALUE.fullmatch(payload):
                raise ValueError(f"Invalid register initialization value {payload!r}")
            base = {"0b": 2, "0x": 16}.get(payload[:2], 10)
            digits = (payload[2:] if base != 10 else payload).replace("_", "")
            value = int(digits or "0", base)
            if randomize and value == 0 and digits and base != 10:
                raise ValueError("Random initialization modulus is zero")
        assignments.append(
            RegisterInitialization(
                circuit.registers[register].members, value, randomize
            )
        )
    return tuple(assignments)
