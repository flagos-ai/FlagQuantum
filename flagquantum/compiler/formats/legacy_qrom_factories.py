# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 data_loading/qrom.py and qrom_base.py.

"""Resolve serialized XOR lookup metadata through the existing QROM owner."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...core.value_parameters import is_symbolic_value
from ...errors import SerializationError
from ..registers.lookup.qrom_component import qrom_program
from ..registers.lookup.qrom_declaration import QromDeclaration
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _dimensions(value: object) -> tuple[SymbolicInt, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise SerializationError("QROM dimension metadata requires a vector")
    values = array.tolist()
    if any(
        not isinstance(dimension, int) and not is_symbolic_value(dimension)
        for dimension in values
    ):
        raise SerializationError("Serialized QROM dimensions require integers")
    return tuple(values)


def read_qrom_declaration(
    arguments: Mapping[str, ResolvedArgument],
) -> QromDeclaration:
    """Read original array metadata for QROM and its blocked source consumers."""
    required = {"data_or_shape", "selection_bitsizes", "target_bitsizes"}
    if not required <= arguments.keys() or arguments.keys() - required - {
        "target_shapes",
        "num_controls",
    }:
        raise SerializationError("Invalid QROM constructor arguments")
    data = arguments["data_or_shape"]
    if (
        not isinstance(data, (np.ndarray, tuple, list))
        or isinstance(data, np.ndarray)
        and data.ndim == 0
    ):
        raise SerializationError("Serialized QROM requires concrete datasets")
    widths = _dimensions(arguments["target_bitsizes"])
    selection = _dimensions(arguments["selection_bitsizes"])
    raw_shapes = arguments.get("target_shapes")
    targets: tuple[tuple[SymbolicInt, ...], ...]
    if raw_shapes is None:
        targets = ((),) * len(widths)
    elif isinstance(raw_shapes, (np.ndarray, tuple, list)):
        targets = tuple(_dimensions(shape) for shape in raw_shapes)
    else:
        raise SerializationError("QROM target shapes require nested vectors")
    if len(targets) != len(widths):
        raise SerializationError("QROM requires one target shape per dataset")
    arrays = tuple(np.asarray(row) for row in data)
    raw_controls = arguments.get("num_controls", 0)
    controls = (
        raw_controls
        if isinstance(raw_controls, bool)
        else _integer({"num_controls": raw_controls}, "num_controls")
    )
    return QromDeclaration.from_arrays(
        *arrays,
        selection_bitsizes=selection,
        target_bitsizes=widths,
        target_shapes=targets,
        num_controls=controls,
    )


def _qrom_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    declaration = read_qrom_declaration(arguments)
    ports = tuple(declaration.signature)
    if registers is not None and ports != registers:
        raise SerializationError("Invalid QROM register signature")
    try:
        program = qrom_program(declaration)
    except (ValueError, TypeError, OverflowError, IndexError):
        # The original library retains literal arrays even if its layout or
        # per-load int conversion fails. Preserve ports, not an invented body.
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(program, name)


def qrom_factories() -> dict[str, ComponentFactory]:
    """Supply the complete concrete-data constructor without duplicating lookup.

    Locked source serialization cannot encode Shaped or symbolic object arrays.
    Native QromDeclaration owns their separate shape-only metadata.
    """
    return {"qualtran.bloqs.data_loading.qrom.QROM": _qrom_factory}
