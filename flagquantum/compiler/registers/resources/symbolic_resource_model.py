# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Add, QROM and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py

"""Shape-level resource formulas without data tables or emitted circuits."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import prod
from typing import cast

from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.gate_budget import GateCategories
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_model import estimate_component_resources
from ..._component_trace.symbolic_math import SymbolicInt, bit_length, smax
from ..arithmetic.arithmetic_declarations import arithmetic_callees
from ..arithmetic.arithmetic_resource_model import source_add_cost
from ..control.control_predicate_resource_model import control_predicate_cost


@dataclass(frozen=True)
class SymbolicResourceFormula:
    """Gate categories and logical width under a source formula."""

    and_bloq: SymbolicInt
    clifford: SymbolicInt
    measurement: SymbolicInt
    logical_qubits: SymbolicInt
    t: SymbolicInt = 0
    toffoli: SymbolicInt = 0
    cswap: SymbolicInt = 0
    rotation: SymbolicInt = 0


def source_add_resource_formula(
    left_width: SymbolicInt,
    right_width: SymbolicInt,
    *,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Use the locked Add declaration and width fallback in the selected profile."""
    if type(legacy_shims) is not bool:
        raise ValueError("Add formula cost profile must be a boolean")
    gates: GateCategories
    peak: SymbolicInt
    if isinstance(left_width, int) and isinstance(right_width, int):
        concrete = source_add_cost(left_width, right_width, legacy_shims=legacy_shims)
        report = estimate_component_resources(concrete)
        gates = report.gates
        peak = report.peak_logical_qubits
    else:
        root = ArithmeticCall("add", (left_width, right_width))
        gates = query_qec_gate_cost(
            root,
            declared_calls={root: arithmetic_callees(root)},
            legacy_shims=legacy_shims,
        )
        signature = Signature(
            (
                Register("a", RegisterType("uint", left_width)),
                Register("b", RegisterType("uint", right_width)),
            )
        )
        peak = cast(SymbolicInt, smax(3, signature.n_qubits()))
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{
            field.name: getattr(gates, field.name)
            for field in fields(SymbolicResourceFormula)
            if field.name != "logical_qubits"
        },
    )


def source_qrom_resource_formula(
    shape: tuple[SymbolicInt, ...],
    target_widths: tuple[SymbolicInt, ...],
    *,
    target_shapes: tuple[tuple[SymbolicInt, ...], ...] | None = None,
    control_count: SymbolicInt = 0,
    selection_widths: tuple[SymbolicInt, ...] | None = None,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Use the locked QROM formula for a shape without concrete table data."""
    if not target_widths:
        raise ValueError("A QROM formula requires at least one target")
    if target_shapes is None:
        target_shapes = ((),) * len(target_widths)
    if len(target_shapes) != len(target_widths):
        raise ValueError("QROM target widths and shapes must have the same length")
    if selection_widths is None:
        selection_widths = tuple(bit_length(extent - 1) for extent in shape)
    elif len(selection_widths) != len(shape):
        raise ValueError("QROM selection widths must match the data shape")

    rows = prod(shape)
    word_widths = tuple(
        width * prod(target_shape)
        for width, target_shape in zip(target_widths, target_shapes, strict=True)
    )
    ands = rows - 2 + control_count
    # Preserve the shape-only source formula, which multiplies target widths.
    loads = rows * prod(word_widths)
    width = sum(word_widths) + 2 * sum(selection_widths) + 2 * control_count - 1
    cleanup = control_predicate_cost((1, 1), uncompute=True, legacy_shims=legacy_shims)
    return SymbolicResourceFormula(
        ands, loads + cleanup.gates.clifford * ands, ands, width
    )


def source_phase_gradient_lookup_resource_formula(
    rows: SymbolicInt,
    selection_width: SymbolicInt,
    phase_width: SymbolicInt,
    control_count: SymbolicInt,
) -> SymbolicResourceFormula:
    """Model one shape-only QROM, add, and QROM sequence."""
    qrom = source_qrom_resource_formula(
        (rows,),
        (phase_width,),
        selection_widths=(selection_width,),
        control_count=control_count,
    )
    address = selection_width + control_count
    addition_peak = smax(3, 2 * phase_width) + address
    width = cast(
        SymbolicInt,
        smax(
            0,
            address + 2 * phase_width,
            qrom.logical_qubits + phase_width,
            addition_peak,
        ),
    )
    return SymbolicResourceFormula(
        2 * qrom.and_bloq,
        2 * qrom.clifford,
        2 * qrom.measurement,
        width,
        toffoli=phase_width - 2,
    )


def source_permutation_cycle_resource_formula(
    cycle_length: SymbolicInt, width: SymbolicInt
) -> SymbolicResourceFormula:
    """Model a shape-only source cycle using unknown equality and XOR words."""
    if isinstance(width, int) and width < 1:
        raise ValueError("Permutation cycle requires a nonempty register")
    if isinstance(cycle_length, int) and cycle_length < 1:
        raise ValueError("Permutation cycle requires at least one element")
    equalities = cycle_length + 1
    ands = equalities * (width - 1)
    # The source decomposes the equality at concrete widths. A symbolic width
    # instead uses the nested signature/callee fallback.
    peak = (
        2 * width
        if isinstance(width, int)
        else cast(SymbolicInt, smax(3, width + 1, 2 * width - 1))
    )
    return SymbolicResourceFormula(
        ands, (equalities + cycle_length) * width, ands, peak
    )
