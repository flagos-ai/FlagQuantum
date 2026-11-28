# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanEqual.build_call_graph and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Shared concrete and symbolic costs for the original unsigned comparison tree."""

from typing import cast

from ....core.value_parameters import is_symbolic_value
from ..._component_trace.qec_gate_cost import QECGateCost
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..control.control_predicate_resource_model import control_predicate_cost
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .less_equal import (
    less_equal_requirements,
    source_equal_width_less_equal_cost,
    source_equal_width_less_equal_gate_cost,
)
from .less_than import source_less_than_constant_cost


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


def source_less_equal_gate_cost(
    left_width: SymbolicInt, right_width: SymbolicInt, *, legacy_shims: bool = False
) -> QECGateCost:
    """Query literal categories, including zero-word declarations without a body.

    Concrete positive words retain the existing tree model. A zero word still
    has the source's negative mixer multiplicity; no emitted gate is implied.
    """
    if not isinstance(left_width, int) or not isinstance(right_width, int):
        return QECGateCost().plus(
            source_less_equal_resource_formula(
                left_width, right_width, legacy_shims=legacy_shims
            ),
            1,
        )
    if min(left_width, right_width) < 0:
        raise ValueError("Comparison categories require nonnegative widths")
    if min(left_width, right_width) > 0:
        return QECGateCost().plus(
            source_less_equal_cost(
                left_width, right_width, legacy_shims=legacy_shims
            ).gates,
            1,
        )
    core = source_equal_width_less_equal_gate_cost(0, legacy_shims=legacy_shims)
    gap = max(left_width, right_width)
    if gap == 0:
        return core
    prefix = QECGateCost(clifford=4)
    if gap > 1:
        prefix = (
            QECGateCost()
            .plus(
                control_predicate_cost((0,) * gap, legacy_shims=legacy_shims).gates, 1
            )
            .plus(
                control_predicate_cost(
                    (0,) * gap, uncompute=True, legacy_shims=legacy_shims
                ).gates,
                1,
            )
        )
    join = (
        QECGateCost()
        .plus(control_predicate_cost((1, 0), legacy_shims=legacy_shims).gates, 1)
        .plus(
            control_predicate_cost(
                (1, 0), uncompute=True, legacy_shims=legacy_shims
            ).gates,
            1,
        )
    )
    return (
        core
        + prefix
        + join
        + QECGateCost(clifford=-1 + 2 * int(right_width > left_width))
    )


def source_less_than_constant_qec_cost(
    width: SymbolicInt, threshold: SymbolicInt
) -> QECGateCost:
    """Query the source comparator categories for concrete or symbolic fields.

    Symbolic thresholds retain the original worst-case set-bit count of width.
    Concrete operands reuse the existing prefix-comparator model. This category
    query carries no logical-workspace estimate.
    """
    if not is_symbolic_value(width, threshold):
        assert isinstance(width, int) and isinstance(threshold, int)
        if width > 0 and threshold >= 0:
            gates = source_less_than_constant_cost(width, threshold).gates
            return QECGateCost().plus(gates, 1)
        if type(width) is not int or width < 0 or type(threshold) is not int:
            raise ValueError(
                "Comparison categories require nonnegative widths and integer thresholds"
            )
        if threshold >= 1 << width:
            return QECGateCost(clifford=1)
        # The source also quotes empty words and negative constants, despite
        # their failed decomposition. This category query supplies no peak.
    set_bits = width if is_symbolic_value(threshold) else int(threshold).bit_count()
    return QECGateCost(
        and_bloq=width, measurement=width, clifford=3 * (width + set_bits) + 2
    )
