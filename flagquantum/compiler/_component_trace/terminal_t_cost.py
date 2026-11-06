# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 t_counts_from_sigma and TComplexity.rotation_cost.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/t_counts_from_sigma.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/t_complexity_protocol.py

"""Compute the locked precision-sensitive model from terminal multiplicities."""

from collections.abc import Mapping
from math import pi

from .resource_classification import classify_resource_gate
from .resource_generalization import (
    ConjunctionCall,
    ControlledRotationCall,
    FixedTCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
)
from .symbolic_math import SymbolicInt, SymbolicReal, ceil, log2


def rotation_t_cost(precision: SymbolicReal) -> SymbolicInt:
    """Return the source's per-rotation estimate before applying multiplicity."""
    return ceil(1.149 * log2(1.0 / precision) + 9.2)


def estimate_terminal_t_cost(sigma: Mapping[ResourceCall, SymbolicInt]) -> SymbolicInt:
    """Apply the source terminal-count convention to explicit resource keys.

    Exact bitwise controlled swaps use seven T gates in this model. Only
    explicit T gates contribute at T-like angles. Rotation synthesis costs
    are rounded per call before multiplication.
    """
    return sum(count * _unit_t_cost(call) for call, count in sigma.items())


def _unit_t_cost(call: ResourceCall) -> SymbolicInt:
    if isinstance(call, PhaseCall):
        return int(call.opcode == "t")
    if isinstance(call, FixedTCall):
        return 4 if call.opcode == "toffoli" else 7
    if isinstance(call, ConjunctionCall):
        return (
            4
            if isinstance(call.controls, tuple)
            and len(call.controls) == 2
            and not call.adjoint
            else 0
        )
    if classify_resource_gate(call) != "rotation":
        return 0
    if isinstance(call, RotationCall):
        return rotation_t_cost(call.precision)
    if isinstance(call, ControlledRotationCall):
        inner = call.call
        if isinstance(inner, PhaseCall):
            raise ValueError(
                "The source terminal model has no precision for controlled S or T"
            )
        value = inner.parameter
        if not isinstance(value, (int, float)):
            raise NotImplementedError(
                "Symbolic controlled rotations have no source stabilizer verdict"
            )
        exponent = value / pi if inner.opcode in {"rx", "ry", "rz"} else float(value)
        # A single controlled Pauli power is stabilizer at integral exponents.
        if exponent.is_integer():
            return 0
        return rotation_t_cost(inner.precision)
    raise AssertionError("Unexpected rotation resource key")
