# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 data_loading/qrom.py and qrom_base.py.

"""Resolve serialized XOR lookup metadata through the existing QROM owner."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ...core.ir.program import ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.qrom_component import qrom_program
from ..registers.lookup.qrom_table import QromTable
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _dimensions(value: object) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise SerializationError("QROM dimension metadata requires a vector")
    values = array.tolist()
    if any(type(dimension) is not int for dimension in values):
        raise SerializationError("Serialized QROM dimensions require integers")
    return tuple(values)


def _qrom_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
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
        or len(data) == 0
    ):
        raise SerializationError("Serialized QROM requires concrete datasets")
    widths = _dimensions(arguments["target_bitsizes"])
    selection = _dimensions(arguments["selection_bitsizes"])
    raw_shapes = arguments.get("target_shapes")
    targets: tuple[tuple[int, ...], ...]
    if raw_shapes is None:
        targets = ((),) * len(widths)
    elif isinstance(raw_shapes, (np.ndarray, tuple, list)):
        targets = tuple(_dimensions(shape) for shape in raw_shapes)
    else:
        raise SerializationError("QROM target shapes require nested vectors")
    if len(targets) != len(widths):
        raise SerializationError("QROM requires one target shape per dataset")
    table = QromTable.from_data(
        *data,
        target_bitsizes=widths,
        target_shapes=targets,
        num_controls=_integer(
            {"num_controls": arguments.get("num_controls", 0)}, "num_controls"
        ),
    )
    table = replace(table, selection_bitsizes=selection)
    if registers is not None and tuple(table.signature) != registers:
        raise SerializationError("Invalid QROM register signature")
    return _rename_program(qrom_program(table), name)


def qrom_factories() -> dict[str, ComponentFactory]:
    """Supply the complete concrete-data constructor without duplicating lookup.

    Locked source serialization cannot encode Shaped or symbolic object arrays.
    The existing shape-only QromTable declaration remains their native owner.
    """
    return {"qualtran.bloqs.data_loading.qrom.QROM": _qrom_factory}
