# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/qvr.py.

"""Bind the original model-only grid QVR without selecting a phase convention."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.grid_qvr_declaration import GridVariableRotationDeclaration
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _grid_qvr_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if arguments.keys() != {"phi_bitsize"}:
        raise SerializationError("Grid QVR requires only phi_bitsize")
    ports = tuple(
        GridVariableRotationDeclaration(_integer(arguments, "phi_bitsize")).signature
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid grid QVR register signature")
    return FunctionRecord(name, (), registers=ports)


def grid_qvr_factories() -> dict[str, ComponentFactory]:
    """Retain the original bodyless root and explicit caller-owned execution."""
    return {
        "qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation": _grid_qvr_factory
    }
