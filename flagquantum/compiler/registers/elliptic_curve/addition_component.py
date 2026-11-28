# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, ECAdd.

"""Thread the six source stages while retaining logical bystanders."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_control import quantum_control_signature


def elliptic_curve_addition_program(
    signature: Signature, stages: tuple[ProgramRecord, ...]
) -> ProgramRecord:
    """Compose six exact stages without weakening their one-sided promises."""
    if len(stages) != 6:
        raise ValueError("Elliptic-curve addition requires six source stages")
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    for stage in stages:
        inputs = Signature(stage.functions[0].registers or ()).lefts()
        values = {register.name: state.pop(register.name) for register in inputs}
        state.update(builder.call_program(stage, values, entry=stage.functions[0].name))
    return builder.finalize(state)


def controlled_elliptic_curve_addition_program(
    signature: Signature, stages: tuple[ProgramRecord, ...]
) -> ProgramRecord:
    """Thread one positive control separately from the six stages' flag ports.

    Each supplied stage already has its source-controlled signature. The outer
    control may have a different local name at each stage; original LEFT/RIGHT
    flag ownership remains unchanged. No stage body or numerical action is inferred.
    """
    if len(stages) != 6:
        raise ValueError("Elliptic-curve addition requires six source stages")
    controlled = quantum_control_signature(signature, ((quantum_bit_type(), ()),))
    control_name = tuple(controlled)[0].name
    builder = ComponentProgramBuilder(controlled)
    state = dict(builder.inputs)
    control = state.pop(control_name)
    for stage in stages:
        registers = stage.functions[0].registers
        if not registers:
            raise ValueError("Controlled curve stages require typed ports")
        prefix, *data_ports = registers
        if (
            prefix.dtype != quantum_bit_type()
            or prefix.shape_symbolic
            or prefix.side is not RegisterSide.THRU
        ):
            raise ValueError(
                "Controlled curve stage requires a scalar through QBit prefix"
            )
        arguments = {
            register.name: state.pop(register.name)
            for register in Signature(data_ports).lefts()
        }
        outputs = builder.call_program(
            stage, {prefix.name: control, **arguments}, entry=stage.functions[0].name
        )
        control = outputs.pop(prefix.name)
        state.update(outputs)
    return builder.finalize({control_name: control, **state})
