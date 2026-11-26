# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _HalfComparisonBase and LinearDepthHalf*.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: explicit logical outputs and native IR; reuse shared half-comparison.

"""Four source half-comparison relations through input exchange and flag flips."""

from collections.abc import Hashable
from typing import Literal

from ....core.ir import Instruction
from ..._component_trace.qec_gate_cost import QECGateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..arithmetic.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)
from ..bits.bitwise import bitwise_not_ir
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .half_comparison import half_greater_ir


def source_unsigned_half_less_than_calls(
    width: SymbolicInt, *, adjoint: bool = False
) -> dict[Hashable, SymbolicInt]:
    """Retain the source unsigned half-LT model, including the no-MSB adder.

    These are logical calls rather than emitted instructions. Inverse output
    aliasing and physical erasure are governed by the existing component owner.
    """
    return {
        ("bitwise_not", width + 1): 3,
        ("cx",): 1,
        ("out_of_place_add", width + 1, adjoint, False): 1,
    }


def _reverse_operands(relation: str) -> bool:
    if relation not in ("gt", "ge", "lt", "le"):
        raise ValueError("Half comparison relation must be gt, ge, lt or le")
    return relation in ("ge", "lt")


def half_compare_outputs(
    left: tuple[int, ...],
    right: tuple[int, ...],
    difference: tuple[int, ...],
    target: int,
    *,
    relation: Literal["gt", "ge", "lt", "le"] = "gt",
    adjoint: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Bind logical ports for the same validated layout passed to half_compare_ir.

    Preserve the pinned executable port aliases: ge/lt adjoints return a on
    the original right qubits and b on the left, although the physical state
    is restored. This source decomposition differs from its classical model;
    callers must retain these bindings rather than silently assume identity.
    """
    reverse = _reverse_operands(relation)
    if type(adjoint) is not bool:
        raise ValueError("Half comparison adjoint must be boolean")
    if adjoint:
        return {"a": right, "b": left} if reverse else {"a": left, "b": right}
    return {"a": left, "b": right, "c": difference, "target": (target,)}


def half_compare_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    difference: tuple[int, ...],
    target: int,
    signs: tuple[int, ...],
    *,
    relation: Literal["gt", "ge", "lt", "le"] = "gt",
    signed: bool = False,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Compute a comparison and retained difference, or erase the correct pair.

    gt/le retain b-a, while ge/lt retain a-b, modulo 2**(n+1).
    Output/initialization and measured-adjoint contracts are half_greater_ir's.
    Physical input qubits and the original pre/post flip order are preserved.
    Logical ports must be read through half_compare_outputs, including the
    pinned ge/lt adjoint output-alias exchange.
    """
    reverse = _reverse_operands(relation)
    a, b = (right, left) if reverse else (left, right)
    inner = half_greater_ir(
        a,
        b,
        difference,
        target,
        signs,
        signed=signed,
        adjoint=adjoint,
        classical_bits=classical_bits,
    )
    flip = bitwise_not_ir((target,)) if relation in ("ge", "le") else ()
    return (*flip, *inner) if adjoint else (*inner, *flip)


def source_half_comparison_gate_cost(
    width: SymbolicInt, *, signed: bool = False, adjoint: bool = False
) -> QECGateCost:
    """Compose the original widened adder, complements and extension calls."""
    if type(signed) is not bool or type(adjoint) is not bool:
        raise ValueError("Half comparison model modes must be boolean")
    addition = source_out_of_place_add_resource_formula(
        width + 1, include_msb=False, adjoint=adjoint
    )
    return QECGateCost(clifford=3 * (width + 1) + 1 + 4 * signed).plus(addition, 1)


def source_half_comparison_resource_formula(
    width: SymbolicInt, *, signed: bool = False, adjoint: bool = False
) -> SymbolicResourceFormula:
    """Retain the unsigned fallback and concrete signed extension peak.

    The original signed symbolic width query fails in SignExtend. Its gate
    counts remain available separately through source_half_comparison_gate_cost.
    """
    gates = source_half_comparison_gate_cost(width, signed=signed, adjoint=adjoint)
    if signed and type(width) is not int:
        raise ValueError(
            "Original signed symbolic half-comparison peak fails in SignExtend"
        )
    addition = source_out_of_place_add_resource_formula(width + 1, include_msb=False)
    return SymbolicResourceFormula(
        gates.and_bloq,
        gates.clifford,
        gates.measurement,
        smax(width + 1, addition.logical_qubits),
    )
