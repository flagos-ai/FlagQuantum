# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata follows Qualtran v0.7.0 QROMBase and blocked lookup constructors.

"""Read shared blocked lookup data, widths, shapes, controls and block logs."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ...errors import SerializationError
from ..registers.lookup.qrom_table import QromTable
from .legacy_component_resolution import ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_qrom_factories import _dimensions


def read_blocked_lookup_arguments(
    arguments: Mapping[str, ResolvedArgument],
) -> tuple[QromTable, tuple[int, ...]]:
    required = {
        "data_or_shape",
        "selection_bitsizes",
        "target_bitsizes",
        "log_block_sizes",
    }
    if not required <= arguments.keys() or arguments.keys() - required - {
        "target_shapes",
        "num_controls",
    }:
        raise SerializationError("Invalid blocked lookup constructor arguments")
    data = arguments["data_or_shape"]
    if (
        not isinstance(data, (tuple, list, np.ndarray))
        or isinstance(data, np.ndarray)
        and data.ndim == 0
        or len(data) == 0
    ):
        raise SerializationError("Serialized blocked lookup requires concrete datasets")
    widths = _dimensions(arguments["target_bitsizes"])
    raw_shapes = arguments.get("target_shapes", ((),) * len(widths))
    if not isinstance(raw_shapes, (tuple, list, np.ndarray)):
        raise SerializationError("Blocked lookup target shapes require nested vectors")
    shapes = tuple(_dimensions(shape) for shape in raw_shapes)
    table = QromTable.from_data(
        *data,
        target_bitsizes=widths,
        target_shapes=shapes,
        num_controls=_integer(
            {"num_controls": arguments.get("num_controls", 0)}, "num_controls"
        ),
    )
    table = replace(
        table, selection_bitsizes=_dimensions(arguments["selection_bitsizes"])
    )
    return table, _dimensions(arguments["log_block_sizes"])
