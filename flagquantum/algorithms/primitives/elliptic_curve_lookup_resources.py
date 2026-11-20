# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECWindowAddR.qrom.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add_r.py

"""Lookup resource declarations for windowed affine point addition."""

from dataclasses import fields
from typing import cast

from ...compiler._component_trace.resource_model import estimate_component_resources
from ...compiler.registers.lookup.qroam_resource_model import (
    source_qroam_clean_cost,
    source_qroam_measurement_erase_cost,
)
from ...compiler.registers.lookup.symbolic_qroam_resource_model import (
    source_blocked_lookup_resource_formula,
    source_shape_erasure_resource_formula,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.value_math import SymbolicInt
from ...core.value_parameters import is_symbolic_value
from .elliptic_curve_point import ECPoint
from .elliptic_curve_window_addition import windowed_point_table


def windowed_point_lookup_resources(
    width: SymbolicInt, window: SymbolicInt, point: ECPoint
) -> dict[str, SymbolicResourceFormula]:
    """Declare the forward and erase lookup costs, including symbolic widths.

    The curve-addition child has its own resource owner. Concrete tables retain
    their data-dependent lookup model. A symbolic word with a concrete window
    uses the original assumption that the word is wider than the window.
    """
    if is_symbolic_value(window):
        raise ValueError(
            "Symbolic point-lookup windows have no constructible source declaration"
        )
    if is_symbolic_value(width):
        shape = (2**window,)
        widths = (width,) * 3
        logs = (0,)
        return {
            "QROAMClean": source_blocked_lookup_resource_formula(
                shape, widths, logs, selection_widths=(window,)
            ),
            "QROAMCleanAdjointWrapper": source_shape_erasure_resource_formula(
                shape, widths, logs
            ).wrapper,
        }

    concrete_width, concrete_window = cast(int, width), cast(int, window)
    table = windowed_point_table(concrete_width, concrete_window, point)
    costs = (
        source_qroam_clean_cost(table, 0, (concrete_window,), (concrete_width,) * 3),
        source_qroam_measurement_erase_cost(table, 0, (concrete_width,) * 3).wrapper,
    )
    result: dict[str, SymbolicResourceFormula] = {}
    for cost in costs:
        report = estimate_component_resources(cost)
        result[cost.name] = SymbolicResourceFormula(
            logical_qubits=report.peak_logical_qubits,
            **{
                field.name: getattr(report.gates, field.name)
                for field in fields(SymbolicResourceFormula)
                if field.name != "logical_qubits"
            },
        )
    return result
