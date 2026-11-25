# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/kinetic.py.

"""Reconstruct the literal source kinetic call chain from shallow metadata."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.arithmetic.grid_kinetic_signature import grid_kinetic_signature
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.layout.component_workspace import free_word_function
from ..registers.phase.grid_qvr_declaration import GridVariableRotationDeclaration
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_product_model_factories import product_model_factories


def _grid_kinetic_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"num_elec", "num_grid"}:
        raise SerializationError("Grid kinetic requires num_elec and num_grid")
    electrons, grid = _integer(arguments, "num_elec"), _integer(arguments, "num_grid")
    if type(grid) is not int:
        raise SerializationError(
            "Grid kinetic bit-length sizing requires a concrete grid"
        )
    assert isinstance(grid, int)
    signature = grid_kinetic_signature(electrons, grid)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid grid kinetic register signature")
    if type(electrons) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    assert isinstance(electrons, int)
    if electrons == 0:
        builder = ComponentProgramBuilder(signature, name=name)
        return builder.finalize(builder.inputs)
    coordinate = signature.get_left("system").bitsize
    assert isinstance(coordinate, int)
    result = 2 * coordinate + 2
    square = product_model_factories()[
        "qualtran.bloqs.arithmetic.multiplication.SumOfSquares"
    ](name + "_square", None, {"bitsize": coordinate, "k": 3})
    assert isinstance(square, FunctionRecord)
    phase = FunctionRecord(
        name + "_phase",
        (),
        registers=tuple(GridVariableRotationDeclaration(result).signature),
    )
    release = free_word_function(name + "_free", opaque_register_type(result), False)
    builder = ComponentProgramBuilder(signature, (square, phase, release), name=name)
    system = np.asarray(builder.inputs["system"], dtype=object)
    for i in range(electrons):
        values = builder.call_dict(square.name, {"input": system[i]})
        system[i] = values["input"]
        word = builder.call_dict(phase.name, {"phi": values["result"]})["phi"]
        builder.call_dict(release.name, {"reg": word})
    return builder.finalize({"system": system})


def grid_kinetic_factories() -> dict[str, ComponentFactory]:
    """Keep the original Free and unspecified-angle QVR in every finite recipe."""
    return {
        "qualtran.bloqs.chemistry.trotter.grid_ham.kinetic.KineticEnergy": _grid_kinetic_factory
    }
