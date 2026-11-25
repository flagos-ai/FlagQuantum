# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/ising/unitaries.py.

"""Store ordered Ising rotations and periodic edges as typed component calls."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def ising_program(
    sites: int, rotation: FunctionRecord, *, cnot: FunctionRecord | None = None
) -> ProgramRecord:
    """Split an MSB-first system word, apply source calls, and join it again.

    With a CNOT dependency each site contributes one directed periodic edge,
    including both edges for two sites. No workspace or measurements are added.
    Dependency definitions own angle lowering and primitive execution.
    """
    if type(sites) is not int or sites < (1 if cnot is None else 2):
        raise ValueError("Ising body requires distinct live edge operands")
    signature = Signature.build(system=sites)
    builder = ComponentProgramBuilder(
        signature, (rotation,) if cnot is None else (rotation, cnot)
    )
    system = builder.inputs["system"]
    assert isinstance(system, RegisterBinding)
    bits = builder.split(system)
    for site in range(sites):
        target = site if cnot is None else (site + 1) % sites
        if cnot is not None:
            pair = builder.call_dict(
                cnot.name, {"ctrl": bits[site], "target": bits[target]}
            )
            bits[site], bits[target] = pair["ctrl"], pair["target"]
        bits[target] = builder.call_dict(rotation.name, {"q": bits[target]})["q"]
        if cnot is not None:
            pair = builder.call_dict(
                cnot.name, {"ctrl": bits[site], "target": bits[target]}
            )
            bits[site], bits[target] = pair["ctrl"], pair["target"]
    return builder.finalize(
        {"system": builder.join(bits, signature.get_left("system").dtype)}
    )
