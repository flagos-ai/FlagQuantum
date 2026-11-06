# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 resource gate predicates.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/classify_bloqs.py

"""Classify declared resource gate keys using the locked source thresholds."""

from math import pi
from typing import Literal

from .resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    _inside_partition,
)

_CLIFFORD_EXPONENTS = (1.0, 0.5, 1.5, -1.0, -0.5, -1.5, 0.0, -2.0, 2.0)
_CLIFFORD_ANGLES = tuple(exponent * pi for exponent in _CLIFFORD_EXPONENTS)
_T_EXPONENTS = (0.25, -0.25)
_T_ANGLES = (pi / 4, -pi / 4)
_ANGLE_ATOL = 1e-12


def classify_resource_gate(
    call: ResourceCall,
) -> Literal["clifford", "t", "rotation"] | None:
    """Recognize a modeled gate family, including the source controlled shim."""
    if isinstance(call, CliffordCall):
        return "clifford"
    if isinstance(call, PhaseCall):
        return "clifford" if call.opcode == "s" else "t"
    if isinstance(call, ControlledRotationCall):
        return "rotation" if len(call.controls) == 1 else None
    if not isinstance(call, RotationCall):
        return None

    value = call.parameter
    if not isinstance(value, (int, float)):
        return "rotation"
    angle = call.opcode in {"rx", "ry", "rz"}
    cliffords = _CLIFFORD_ANGLES if angle else _CLIFFORD_EXPONENTS
    t_values = _T_ANGLES if angle else _T_EXPONENTS
    if any(abs(value - target) < _ANGLE_ATOL for target in cliffords):
        return "clifford"
    if any(abs(value - target) < _ANGLE_ATOL for target in t_values):
        return "t"
    return "rotation"


def ignore_known_cliffords(call: ResourceCall) -> ResourceCall | None:
    """Omit source-recognized Clifford keys, recursing through partitions."""
    if classify_resource_gate(call) == "clifford":
        return None
    return _inside_partition(ignore_known_cliffords, call)
