# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Comparator.build_composite_bloq and its adjoint.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py
# Changes: native IR, explicit logical history and reused clean comparison work.

"""Reversible compare/exchange retaining the original logical swap history."""

from ....core.ir import Instruction
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt
from ..primitives.controlled_swap import controlled_swap_ir
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .comparison_resource_model import source_less_equal_resource_formula
from .ordering import greater_adjoint_ir, greater_ir, source_greater_cost


def compare_exchange_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    history: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Sort two unsigned words and record a>b, or undo a correct output tuple.

    Inputs are equal positive-width big-endian words. Forward history starts
    zero and is logical output, not clean work. Inverse requires min/max and
    their correct original history; it restores both inputs and clears history.
    Shared comparison work starts/ends zero in either direction, using
    less_equal_requirements(n,n). Classical slots retain latest outcomes.
    """
    if type(adjoint) is not bool:
        raise ValueError("Compare/exchange adjoint must be boolean")
    if not left or len(left) != len(right):
        raise ValueError("Compare/exchange requires equal positive operand widths")
    emit = greater_adjoint_ir if adjoint else greater_ir
    comparison = emit(left, right, history, workspace, classical_bits)
    swap = controlled_swap_ir(history, left, right)
    return (*swap, *comparison) if adjoint else (*comparison, *swap)


def source_compare_exchange_cost(width: int, *, adjoint: bool = False) -> ComponentCost:
    """Compose GreaterThan and width CSwap calls under the original model."""
    if type(adjoint) is not bool:
        raise ValueError("Compare/exchange model adjoint must be boolean")
    greater = source_greater_cost(width, width)
    return ComponentCost(
        "ComparatorAdjoint" if adjoint else "Comparator",
        2 * width + 1,
        GateCost(cswap=width),
        (CostCall(greater),),
    )


def source_compare_exchange_resource_formula(
    width: SymbolicInt,
) -> SymbolicResourceFormula:
    """Compose the existing symbolic comparison query with CSwap declarations."""
    comparison = source_less_equal_resource_formula(width, width)
    return SymbolicResourceFormula(
        comparison.and_bloq,
        comparison.clifford + 1,
        comparison.measurement,
        comparison.logical_qubits,
        cswap=width,
    )
