# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Coefficient adaptation follows locked Qualtran v0.7.0 prepare_hubbard.py.

"""Convert Hubbard coefficient payloads and delegate packed Program parsing."""

from math import isfinite

from ..algorithms.primitives.hubbard_prepare import _hubbard_preparation_angles
from ..compiler.formats.legacy_block_encoding_factories import ControlledSelect
from ..compiler.formats.legacy_component_resolution import ComponentFactory
from ..compiler.formats.legacy_hubbard_packed_context import (
    hubbard_packed_oracle_context,
)
from ..compiler.formats.legacy_packed_oracle_factories import OracleContext
from ..core.value_math import SymbolicReal
from ..errors import SerializationError


def _prepare_angles(
    size: int, t: SymbolicReal, u: SymbolicReal
) -> tuple[float, float] | None:
    if not isinstance(t, (int, float)) or not isinstance(u, (int, float)):
        return None
    values = _hubbard_preparation_angles(size, t, u)
    if not all(isfinite(value) for value in values):
        raise SerializationError("Hubbard preparation requires finite source angles")
    return values


def _packed_oracle_library_context(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
) -> tuple[dict[str, ComponentFactory], OracleContext, OracleContext, ControlledSelect]:
    """Inject coefficient conversion into the shared scoped Program owner."""
    return hubbard_packed_oracle_context(
        angles=_prepare_angles,
        prepare_context=prepare_context,
        select_context=select_context,
    )


def packed_oracle_library_factories(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
) -> dict[str, ComponentFactory]:
    """Supply the original packed constructors through native oracle owners."""
    return _packed_oracle_library_context(
        prepare_context=prepare_context, select_context=select_context
    )[0]
