# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ArcTan.apply contract from locked Qualtran v0.7.0 arithmetic/trigonometric/arctan.py.

"""Store the existing bounded arctangent lookup as a composable native Program."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .arctan_lookup import _arctan_magnitude_words, arctan_lookup_requirements
from .arctangent_declaration import ArctangentDeclaration
from .qrom_component import qrom_program
from .qrom_table import QromTable


def arctan_lookup_program(
    source_width: int, target_width: int, *, controlled: bool = False
) -> ProgramRecord:
    """XOR the exact bounded table into occupied MSB-first sign and target ports.

    This is the native QROM implementation of the original mathematical map,
    not an upstream ArcTan decomposition: the source provides none. The shared
    QROM owns, measurement-erases and releases its predicate workspace. Inputs
    remain live, and a second lookup erases the first on every measurement
    branch. Bounds and table rounding reuse the existing flat lookup owner.
    """
    if type(controlled) is not bool:
        raise ValueError("Arctangent control mode must be boolean")
    arctan_lookup_requirements(source_width, target_width)
    builder = ComponentProgramBuilder(
        Signature(
            (
                *((Register("ctrl", quantum_bit_type()),) if controlled else ()),
                *ArctangentDeclaration(source_width, target_width).signature,
            )
        )
    )
    if source_width == 0:
        return builder.finalize(builder.inputs)
    signs = (0,) + (1,) * ((1 << source_width) - 1)
    data = (
        (signs, _arctan_magnitude_words(source_width, target_width))
        if target_width
        else (signs,)
    )
    table = QromTable.from_data(
        *data,
        target_bitsizes=(1, target_width) if target_width else (1,),
        num_controls=int(controlled),
    )
    arguments = {
        "selection": builder.inputs["select"],
        "target0_": builder.inputs["sign"],
    }
    if target_width:
        arguments["target1_"] = builder.inputs["target"]
    if controlled:
        arguments["control"] = builder.inputs["ctrl"]
    result = builder.call_program(
        qrom_program(table, reserve_workspace=True),
        arguments,
    )
    outputs = {"select": result["selection"], "sign": result["target0_"]}
    if target_width:
        outputs["target"] = result["target1_"]
    if controlled:
        outputs = {"ctrl": result["control"], **outputs}
    return builder.finalize(outputs)
