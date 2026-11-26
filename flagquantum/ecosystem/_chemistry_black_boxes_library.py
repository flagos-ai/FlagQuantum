# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 chemistry/black_boxes.py.

"""Bind analytical chemistry lookup placeholders without fabricating circuits."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.primitives.chemistry_qroam_model import chemistry_qroam_signature
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_thc_selection_factories import thc_selection_factories
from ..core.ir.program import FunctionRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError

_BASE = "qualtran.bloqs.chemistry.black_boxes."


def _positive(arguments: Mapping[str, ResolvedArgument], field: str) -> int:
    value = arguments.get(field)
    if type(value) is not int or value < 1:
        raise SerializationError(
            f"Chemistry QROAM {field} requires a positive concrete integer"
        )
    return value


def _lookup(
    two_axes: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    required = (
        {
            "data_a_size",
            "data_b_size",
            "data_a_block_size",
            "data_b_block_size",
            "target_bitsize",
        }
        if two_axes
        else {"data_size", "target_bitsize"}
    )
    optional = {"is_adjoint"} | (set() if two_axes else {"qroam_block_size"})
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid chemistry QROAM constructor fields")
    sizes = {field: _positive(arguments, field) for field in required}
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int) or direction not in (0, 1):
        raise SerializationError("Chemistry QROAM direction requires a binary flag")
    if arguments.get("qroam_block_size") is not None:
        _positive(arguments, "qroam_block_size")
    rows = sizes["data_a_size" if two_axes else "data_size"]
    ports = tuple(chemistry_qroam_signature(rows, sizes["target_bitsize"]))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid chemistry QROAM source register signature")
    return FunctionRecord(name, (), registers=ports)


def chemistry_black_boxes_factories() -> dict[str, ComponentFactory]:
    """Resolve original analytical placeholders and reuse the owned phase wrapper."""
    return {
        _BASE + "QROAM": partial(_lookup, False),
        _BASE + "QROAMTwoRegs": partial(_lookup, True),
        _BASE
        + "ApplyControlledZs": thc_selection_factories()[_BASE + "ApplyControlledZs"],
    }
