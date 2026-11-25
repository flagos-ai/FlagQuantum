# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ArcTan.apply contract from locked Qualtran v0.7.0 arithmetic/trigonometric/arctan.py.

"""Store the existing bounded arctangent lookup as a composable native Program."""

from ....core.ir.program import ProgramRecord
from ..component_builder import ComponentProgramBuilder
from .arctan_lookup import _arctan_magnitude_words, arctan_lookup_requirements
from .arctangent_declaration import ArctangentDeclaration
from .qrom_component import qrom_program
from .qrom_table import QromTable


def arctan_lookup_program(source_width: int, target_width: int) -> ProgramRecord:
    """XOR the exact bounded table into occupied MSB-first sign and target ports.

    This is the native QROM implementation of the original mathematical map,
    not an upstream ArcTan decomposition: the source provides none. The shared
    QROM owns, measurement-erases and releases its predicate workspace. Inputs
    remain live, and a second lookup erases the first on every measurement
    branch. Bounds and table rounding reuse the existing flat lookup owner.
    """
    arctan_lookup_requirements(source_width, target_width)
    signs = (0,) + (1,) * ((1 << source_width) - 1)
    table = QromTable.from_data(
        signs,
        _arctan_magnitude_words(source_width, target_width),
        target_bitsizes=(1, target_width),
    )
    builder = ComponentProgramBuilder(
        ArctangentDeclaration(source_width, target_width).signature
    )
    result = builder.call_program(
        qrom_program(table, reserve_workspace=True),
        {
            "selection": builder.inputs["select"],
            "target0_": builder.inputs["sign"],
            "target1_": builder.inputs["target"],
        },
    )
    return builder.finalize(
        {
            "select": result["selection"],
            "sign": result["target0_"],
            "target": result["target1_"],
        }
    )
