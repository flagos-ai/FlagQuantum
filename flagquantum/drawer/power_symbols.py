# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Power._circuit_diagram_info_.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Decorate resolved child labels for a repeated register operation."""

from collections.abc import Sequence

from ..core.value_math import SymbolicInt


def power_wire_symbols(
    child_symbols: Sequence[str], repetitions: SymbolicInt
) -> tuple[str, ...]:
    """Append the source exponent suffix to each label in wire order."""
    return tuple(f"{symbol}^{repetitions}" for symbol in child_symbols)
