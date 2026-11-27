# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 QROM and UnaryIterationGate call graphs.

"""Source QROM declared calls without requiring a dense physical table."""

from collections import Counter
from operator import index
from typing import cast

from ....core.ir import Instruction
from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..._component_trace.resource_generalization import (
    CliffordCall,
    ControlPredicateCall,
)
from ..arithmetic.arithmetic_declarations import arithmetic_call_graph
from .qrom_data import qrom_break_early, qrom_load_words
from .qrom_declaration import QromDeclaration
from .qrom_resource_model import QromCall
from .unary_iteration import _walk_bounded_unary_tree


def declaration_qrom_calls(declaration: QromDeclaration) -> dict[QromCall, int]:
    """Quote source calls excluding Split/Join, retaining zero-count edges.

    This query can succeed when physical loads fail: nonconstant negative words
    contribute their Python bit count, while zero-width XorK calls remain
    declarations. Unknown-data formulae belong to source_qrom_resource_formula.
    """
    data = declaration.data
    selectors = tuple(
        reg for reg in declaration.signature if reg.name.startswith("selection")
    )
    widths = tuple(index(reg.bitsize) for reg in selectors)
    extents = tuple(index(n) for n in declaration.selection_axes())
    controls = index(declaration.num_controls)
    targets = cast(tuple[tuple[int, ...], ...], declaration.target_shapes)
    calls: Counter[QromCall] = Counter()

    def reduction() -> None:
        if controls > 1:
            name = "And" if controls == 2 else "MultiAnd"
            calls[(name, (1,) * controls, False)] += 1
            calls[(name, (1,) * controls, True)] += 1

    if not sum(widths) or qrom_break_early(data, (), 0, extents[0]):
        # Source costing takes its Composite bridge on this branch. Empty
        # register arrays fail that bridge even if the Cirq generator is empty.
        if declaration.is_symbolic():
            raise ValueError(
                "Source zero-selection graph cannot decompose parameterized metadata"
            )
        if controls < 0:
            raise ValueError(
                "Source zero-selection bridge cannot bind negative controls"
            )
        if any(0 in shape for shape in targets):
            raise ValueError(
                "Cannot apply_along_axis when any iteration dimensions are 0"
            )
        reduction()
        for dataset, _, value in qrom_load_words(
            data, targets, (0,) * len(declaration.data_shape)
        ):
            calls[
                (
                    "XorK",
                    (
                        index(declaration.target_bitsizes[dataset]),
                        value,
                        min(1, controls),
                    ),
                    False,
                )
            ] += 1
        return dict(calls)
    reduction()

    def emit_gate(gate: Instruction) -> None:
        name = "X" if gate.name == "x" else "CNOT"
        calls[(name, (), False)] += 1

    def emit_and(
        left: int, right: int, target: int, values: tuple[int, int], erase: bool
    ) -> None:
        calls[("And", values, erase)] += 1

    def visit(level: int, prefix: tuple[int, ...]) -> None:
        if level == len(widths):
            calls[("CNOT", (), False)] += sum(
                value.bit_count()
                for _, _, value in qrom_load_words(data, targets, prefix)
            )
            return
        width = widths[level]
        _walk_bounded_unary_tree(
            0 if level or controls else None,
            tuple(range(1, width + 1)),
            extents[level],
            tuple(range(width + 1, 2 * width + 1)),
            lambda selected, _: visit(level + 1, prefix + (selected,)),
            emit_gate,
            emit_and,
            break_early=lambda start, stop: qrom_break_early(data, prefix, start, stop),
        )

    visit(0, ())
    # The source segment-tree counter retains zero-count And/CNOT keys.
    for key in (("And", (1, 0), False), ("And", (1, 1), True), ("CNOT", (), False)):
        calls[key] += 0
    return dict(calls)


def declaration_qrom_gate_cost(declaration: QromDeclaration) -> QECGateCost:
    """Aggregate existing typed child models, separately from primitive counts."""
    total = QECGateCost()
    for (name, parameters, adjoint), repetitions in declaration_qrom_calls(
        declaration
    ).items():
        if name == "XorK":
            width, constant, controls = parameters
            child = ArithmeticCall(
                "controlled_xor_constant" if controls else "xor_constant",
                (width,),
                constant=constant,
            )
            cost = query_qec_gate_cost(
                child, declared_calls=arithmetic_call_graph(child)
            )
        elif name in {"And", "MultiAnd"}:
            cost = query_qec_gate_cost(
                ControlPredicateCall(
                    tuple(index(value) for value in parameters), adjoint
                )
            )
        else:
            cost = query_qec_gate_cost(
                CliffordCall("x", 1) if name == "X" else CliffordCall("cx", 2)
            )
        total = total.plus(cost, repetitions)
    return total
