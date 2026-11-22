# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix py/val/angle.pybind.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Convert external Python phase values into the exact owned full-turn ring."""

from fractions import Fraction
from typing import Any

from ..compiler.artifacts.phase_angle import PhaseAngle128


def phase_angle_from_half_turns(value: Any) -> PhaseAngle128:
    """Normalize through Fraction and round to the exact 128-bit phase grid.

    Fraction owns external-object admission and its conversion exceptions.
    The returned phase is independent of the caller's original object.
    """
    normalized = Fraction(value) % 2
    rounded = round(normalized * (1 << 127))
    return PhaseAngle128._from_units(rounded)
