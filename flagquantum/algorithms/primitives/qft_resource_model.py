# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the declared graph of Qualtran v0.7.0 QFTTextBook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/qft_text_book.py

"""Textbook Fourier transform declarations independent of circuit execution."""

from dataclasses import replace

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, smax
from ...compiler.registers.phase.phase_gradient import (
    source_phase_gradient_unitary_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature


def source_textbook_qft_signature(width: SymbolicInt) -> Signature:
    """Describe the single through unsigned register in the source."""
    return Signature((Register("q", unsigned_register_type(width)),))


def source_textbook_qft_calls(
    width: SymbolicInt, *, with_reverse: bool = True
) -> dict[tuple[str, SymbolicInt], SymbolicInt]:
    """Declare Hadamard, controlled phase-gradient and optional reversal calls."""
    calls: dict[tuple[str, SymbolicInt], SymbolicInt] = {("Hadamard", 1): width}
    if type(width) is int:
        for bits in range(1, width):
            calls[("PhaseGradientUnitary", bits)] = 1
    else:
        calls[("PhaseGradientUnitary", width - 1)] = width // 2
    if with_reverse:
        calls[("TwoBitSwap", 2)] = width // 2
    return calls


def source_textbook_qft_cost(width: int, *, with_reverse: bool = True) -> ComponentCost:
    """Reuse the source controlled phase-gradient cost at each concrete width."""
    if type(width) is not int or width < 0:
        raise ValueError("Textbook Fourier cost requires a nonnegative integer width")
    calls = [CostCall(ComponentCost("Hadamard", 1, GateCost(clifford=1)), width)]
    for bits in range(1, width):
        gradient = source_phase_gradient_unitary_cost(bits, 0.5, controlled=True)
        calls.append(
            CostCall(
                replace(gradient, name=f"PhaseGradientUnitary({bits})"),
                bystander_qubits=width - bits - 1,
            )
        )
    if with_reverse:
        calls.append(
            CostCall(ComponentCost("TwoBitSwap", 2, GateCost(clifford=1)), width // 2)
        )
    return ComponentCost("QFTTextBook", width, calls=tuple(calls))


def source_textbook_qft_symbolic_formula(
    width: SymbolicInt, *, with_reverse: bool = True
) -> SymbolicResourceFormula:
    """Evaluate the locked source's grouped symbolic phase-gradient rule."""
    if isinstance(width, int):
        raise TypeError("Use the concrete textbook Fourier model for an integer width")
    half = width // 2
    phases = (width - 1) * half
    swaps = half if with_reverse else 0
    return SymbolicResourceFormula(
        and_bloq=phases,
        clifford=width + phases + swaps,
        measurement=phases,
        rotation=phases,
        logical_qubits=smax(3, width),
    )
