# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Addition and GF2AddK.build_call_graph.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic

"""Field XOR declarations without numerical field objects or gate emission."""

from __future__ import annotations

from operator import index
from typing import SupportsIndex

from ....core.value_parameters import is_symbolic_value
from ..._component_trace.resource_generalization import CliffordCall
from ..._component_trace.symbolic_math import SymbolicInt
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def field_addition_calls(width: SymbolicInt) -> dict[CliffordCall, SymbolicInt]:
    """Declare the locked register addition's width CNOT calls."""
    return {CliffordCall("cx", 2): width}


def field_add_constant_calls(
    width: SymbolicInt, constant: SymbolicInt
) -> dict[CliffordCall, SymbolicInt]:
    """Retain zero-count X edges and the source's symbolic width upper bound."""
    count = width
    if not is_symbolic_value(width, constant):
        if not isinstance(width, SupportsIndex) or index(width) < 1:
            raise ValueError("Field addition requires a positive integer width")
        if not isinstance(constant, SupportsIndex):
            raise ValueError("Field constant must fit the declared width")
        value = index(constant)
        if not 0 <= value < 1 << index(width):
            raise ValueError("Field constant must fit the declared width")
        count = value.bit_count()
    return {CliffordCall("x", 1): count}


def field_addition_resource_formula(
    width: SymbolicInt, *, constant: SymbolicInt | None = None
) -> SymbolicResourceFormula:
    """Separate source Clifford counts and logical width from emitted primitives.

    An omitted constant denotes two-register addition. A symbolic constant
    follows the unchanged source bound even when its value is known to be zero.
    Both operations require zero auxiliary wires and zero measurements.
    """
    calls = (
        field_addition_calls(width)
        if constant is None
        else field_add_constant_calls(width, constant)
    )
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=sum(calls.values()),
        measurement=0,
        logical_qubits=(2 if constant is None else 1) * width,
    )
