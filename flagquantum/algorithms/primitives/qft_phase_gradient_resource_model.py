# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QFTPhaseGradient.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/qft_phase_gradient.py

"""Declared phase-gradient Fourier resources and register metadata."""

from dataclasses import dataclass
from typing import TypeAlias

from sympy import log

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.resource_generalization import CliffordCall
from ...compiler._component_trace.symbolic_math import SymbolicInt, smax
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature


@dataclass(frozen=True)
class PhaseGradientQFTCall:
    width: SymbolicInt
    with_reverse: bool = True


@dataclass(frozen=True)
class ProductAccumulationCall:
    left: SymbolicInt
    right: SymbolicInt
    result: SymbolicInt


PhaseGradientQFTNode: TypeAlias = (
    PhaseGradientQFTCall | ProductAccumulationCall | CliffordCall
)
_HADAMARD = CliffordCall("h", 1)
_SWAP = CliffordCall("swap_bit", 2)


def source_phase_gradient_qft_signature(width: SymbolicInt) -> Signature:
    """Declare the unsigned data and fractional gradient through registers."""
    return Signature(
        (
            Register("q", unsigned_register_type(width)),
            Register("phase_grad", fixed_point_register_type(width, width)),
        )
    )


def source_phase_gradient_qft_calls(
    node: PhaseGradientQFTNode,
) -> dict[PhaseGradientQFTNode, SymbolicInt]:
    """Preserve the source declaration, including equal recursive child keys."""
    if not isinstance(node, PhaseGradientQFTCall):
        return {}
    width = node.width
    if type(width) is int and width < 1:
        raise ValueError("Phase-gradient Fourier declarations require positive width")
    left = width // 2
    right = width - left
    product = ProductAccumulationCall(left, right, width)
    if not isinstance(width, int):
        return {product: log(width, 2)}
    if width == 1:
        return {_HADAMARD: 1}
    calls: dict[PhaseGradientQFTNode, SymbolicInt] = {
        PhaseGradientQFTCall(left): 1,
        PhaseGradientQFTCall(right): 1,
        product: 1,
    }
    if node.with_reverse:
        calls[_SWAP] = left
    return calls


def source_phase_gradient_qft_graph(
    width: SymbolicInt, *, with_reverse: bool = True, max_depth: int | None = None
) -> SymbolicCallGraph[PhaseGradientQFTNode]:
    """Expand the locked call model without constructing a circuit."""
    root = PhaseGradientQFTCall(width, with_reverse)
    return symbolic_call_graph(
        root, lambda node, _: source_phase_gradient_qft_calls(node), max_depth=max_depth
    )


def source_phase_gradient_qft_formula(
    width: SymbolicInt, *, with_reverse: bool = True
) -> SymbolicResourceFormula:
    """Use the source product-leaf estimate and source aggregate counts."""
    graph = source_phase_gradient_qft_graph(width, with_reverse=with_reverse)
    clifford: SymbolicInt = graph.leaf_counts.get(_HADAMARD, 0) + graph.leaf_counts.get(
        _SWAP, 0
    )
    t: SymbolicInt = 0
    for child, count in graph.leaf_counts.items():
        if isinstance(child, ProductAccumulationCall):
            t += count * 8 * smax(child.left, child.right) ** 2
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=clifford,
        measurement=0,
        t=t,
        logical_qubits=smax(1, 2 * width),
    )
