# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute declarations and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Hamming-count resource queries independent of gate construction and phasing."""

from dataclasses import fields
from typing import cast

from .._component_trace.arithmetic_call import ArithmeticCall
from .._component_trace.qec_gate_cost import query_qec_gate_cost
from .._component_trace.resource_model import ComponentCost, GateCost
from .._component_trace.symbolic_math import SymbolicInt, smax
from .arithmetic_declarations import arithmetic_call_graph
from .hamming_weight import hamming_weight_requirements, hamming_weight_resource_widths
from .symbolic_resource_model import SymbolicResourceFormula


def source_hamming_weight_resource_formula(
    width: SymbolicInt, *, adjoint: bool = False, legacy_shims: bool = False
) -> SymbolicResourceFormula:
    """Match original QEC categories and successful qubit-count queries.

    Python zero/one retain failed source width boundaries. Symbolic zero
    retains the original negative declaration counts as diagnostic model data.
    Symbolic widths use the original one-bit population-count lower bound,
    not an evaluated Python population count. No symbolic gates are emitted.
    """
    root = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    if type(legacy_shims) is not bool:
        raise ValueError("Hamming resource profile must be boolean")
    if type(width) is int and width in (0, 1):
        raise ValueError("Original Hamming qubit-count query fails at this width")
    junk, output = hamming_weight_resource_widths(width)
    gates = query_qec_gate_cost(
        root, declared_calls=arithmetic_call_graph(root), legacy_shims=legacy_shims
    )
    peak = width + junk + output
    if type(width) is not int:
        peak = cast(SymbolicInt, smax(3, width, peak))
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{f.name: getattr(gates, f.name) for f in fields(gates)},
    )


def source_hamming_weight_cost(
    width: int, *, adjoint: bool = False, legacy_shims: bool = False
) -> ComponentCost:
    """Query original gates with the explicit native logical register storage.

    Unlike the source QubitCount query, this declaration can describe width
    one's two-wire native implementation. Consumers must report that source
    query failure separately rather than claiming successful peak parity.
    """
    junk, output = hamming_weight_requirements(width)
    root = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    if type(legacy_shims) is not bool:
        raise ValueError("Hamming resource profile must be boolean")
    gates = query_qec_gate_cost(
        root, declared_calls=arithmetic_call_graph(root), legacy_shims=legacy_shims
    )
    return ComponentCost(
        "HammingWeightComputeAdjoint" if adjoint else "HammingWeightCompute",
        width + junk + output,
        GateCost(**{f.name: cast(int, getattr(gates, f.name)) for f in fields(gates)}),
    )
