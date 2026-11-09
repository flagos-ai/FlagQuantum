# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanEqual.build_call_graph and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Shared concrete and symbolic costs for the original unsigned comparison tree."""

from typing import cast

from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..control.control_predicate_resource_model import control_predicate_cost
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .less_equal import less_equal_requirements, source_equal_width_less_equal_cost


def source_less_equal_cost(
    left_width: int, right_width: int, *, legacy_shims: bool = False
) -> ComponentCost:
    """Retain the shorter-word tree, prefix equality and final source conjunction."""
    work, _ = less_equal_requirements(left_width, right_width)
    core = source_equal_width_less_equal_cost(
        min(left_width, right_width), legacy_shims=legacy_shims
    )
    if left_width == right_width:
        return core
    gap = abs(left_width - right_width)
    prefix = GateCost(clifford=4)
    if gap > 1:
        prefix = control_predicate_cost(
            (0,) * gap, legacy_shims=legacy_shims
        ).gates.plus(
            control_predicate_cost(
                (0,) * gap, uncompute=True, legacy_shims=legacy_shims
            ).gates,
            1,
        )
    join = control_predicate_cost((1, 0), legacy_shims=legacy_shims).gates.plus(
        control_predicate_cost((1, 0), uncompute=True, legacy_shims=legacy_shims).gates,
        1,
    )
    return ComponentCost(
        "LessThanEqual",
        left_width + right_width + 1 + work,
        GateCost(
            and_bloq=core.gates.and_bloq + prefix.and_bloq + join.and_bloq,
            # Replace equal-width X,CX by the guarded CX, then retain the
            # source's two extra target flips when the right register is longer.
            clifford=core.gates.clifford
            - 1
            + prefix.clifford
            + join.clifford
            + 2 * (right_width > left_width),
            measurement=core.gates.measurement + prefix.measurement + join.measurement,
        ),
    )


def source_less_equal_resource_formula(
    left_width: SymbolicInt, right_width: SymbolicInt, *, legacy_shims: bool = False
) -> SymbolicResourceFormula:
    """Preserve the original symbolic declaration and signature-width fallback.

    Sympy integer expressions remain symbolic, as in source. The symbolic
    call graph uses left_width alone, even for unequal or partly bound words.
    It is not inferred from the concrete shorter-word algorithm.
    """
    if any(
        isinstance(width, (bool, float)) or (isinstance(width, int) and width < 1)
        for width in (left_width, right_width)
    ):
        raise ValueError(
            "Comparison model widths must be positive integers or symbolic expressions"
        )
    if isinstance(left_width, int) and isinstance(right_width, int):
        model = source_less_equal_cost(
            left_width, right_width, legacy_shims=legacy_shims
        )
        return SymbolicResourceFormula(
            model.gates.and_bloq,
            model.gates.clifford,
            model.gates.measurement,
            model.qubits,
        )
    cleanup = control_predicate_cost((1, 1), uncompute=True, legacy_shims=legacy_shims)
    ands = 2 * left_width + 1
    return SymbolicResourceFormula(
        ands,
        20 * left_width + 10 + cleanup.gates.clifford * ands,
        ands,
        cast(SymbolicInt, smax(7, left_width + right_width + 1)),
    )
