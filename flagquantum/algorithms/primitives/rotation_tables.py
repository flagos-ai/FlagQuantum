# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 RotationTree.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py

"""Concrete angle tables for phase-gradient state preparation."""

from ...compiler.registers.phase.rotation_table_words import rotation_table_words


def rotation_tables(
    coefficients: tuple[complex, ...], phase_width: int, *, uncompute: bool = False
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return the existing compiler's constant big-endian rotation table words."""
    return rotation_table_words(coefficients, phase_width, uncompute=uncompute)
