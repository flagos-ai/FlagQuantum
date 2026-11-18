# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Power._circuit_diagram_info_.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Decorate resolved child labels for a repeated register operation."""

from collections.abc import Sequence

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicInt


def power_wire_symbols(
    child_symbols: Sequence[str] | None,
    repetitions: SymbolicInt,
    *,
    registers: Sequence[Register] = (),
) -> tuple[str, ...]:
    """Suffix resolved child labels or derive the source register fallback.

    A missing child diagram uses the ordered concrete register widths and the
    whole Power title on the first wire. As in the source, an empty fallback
    signature has no first wire and raises IndexError.
    """
    if child_symbols is None:
        labels: list[str] = []
        for register in registers:
            width = register.total_bits()
            if not isinstance(width, int):
                raise TypeError("Power diagram fallback needs concrete register widths")
            labels.extend([register.name] * width)
        labels[0] = "Power"
        child_symbols = labels
    return tuple(f"{symbol}^{repetitions}" for symbol in child_symbols)
