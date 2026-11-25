# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/potential.py.

"""Compose source potential constructors with the existing real coefficient codec."""

from ..algorithms.primitives.grid_inverse_sqrt import _HIGH, _LOW
from ..compiler.formats.legacy_component_resolution import ComponentFactory
from ..compiler.formats.legacy_grid_potential_factories import grid_potential_factories
from .inverse_sqrt_coefficients import inverse_sqrt_coefficient_qrom_data


def _coefficient_data(selection: int, precision: int) -> tuple[tuple[int, ...], ...]:
    return inverse_sqrt_coefficient_qrom_data(selection, precision, (_LOW, _HIGH))


def grid_potential_library_factories() -> dict[str, ComponentFactory]:
    """Convert default coefficients at the ecosystem boundary before construction."""
    return grid_potential_factories(coefficient_data=_coefficient_data)
