# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApproximateQFT's metadata and call graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/approximate_qft.py

"""Locked approximate Fourier declarations independent of emitted gates."""

from typing import TypeAlias

from sympy import pi, sqrt

from ...compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    ceil,
    log2,
    smax,
)
from ...compiler.registers.arithmetic.arithmetic_resource_model import (
    source_phase_add_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .approximate_qft import approximate_qft_precision

ApproximateQFTCall: TypeAlias = tuple[str, tuple[SymbolicInt, ...]]


def source_approximate_qft_default_precision(width: SymbolicInt) -> SymbolicInt:
    """Return the locked default ceil(log2(width))."""
    return ceil(log2(width))


def source_approximate_qft_epsilon_precision(
    width: SymbolicInt, epsilon: SymbolicReal
) -> SymbolicInt:
    """Retain the source symbolic bound and reuse the concrete calculation."""
    if isinstance(width, int) and isinstance(epsilon, (int, float)):
        return approximate_qft_precision(width, epsilon)
    return ceil(log2(width * pi * sqrt(2) / epsilon))


def source_approximate_qft_signature(
    width: SymbolicInt, precision: SymbolicInt
) -> Signature:
    """Declare the unsigned data and fractional-gradient through registers."""
    data = Register("q", unsigned_register_type(width))
    if precision == 0:
        return Signature((data,))
    return Signature(
        (data, Register("phase_grad", fixed_point_register_type(precision, precision)))
    )


def source_approximate_qft_calls(
    width: SymbolicInt, precision: SymbolicInt, *, with_reverse: bool = True
) -> dict[ApproximateQFTCall, SymbolicInt]:
    """Declare the locked controlled shift-one windows, including zero width."""
    calls: dict[ApproximateQFTCall, SymbolicInt] = {}
    if isinstance(width, int) and isinstance(precision, int):
        if width < 1 or precision < int(width > 1):
            raise ValueError("Approximate Fourier declarations require valid widths")
        for i in range(1, width):
            active = min(i, precision - 1)
            key = ("AddIntoPhaseGrad", (active, active + 1))
            calls[key] = calls.get(key, 0) + 1
    else:
        calls[("AddIntoPhaseGrad", (precision, precision))] = width
    calls[("Hadamard", ())] = width
    if with_reverse:
        calls[("TwoBitSwap", ())] = width // 2
    return calls


def source_approximate_qft_formula(
    width: SymbolicInt, precision: SymbolicInt, *, with_reverse: bool = True
) -> SymbolicResourceFormula:
    """Evaluate the source model, retaining its diagnostic negative estimate."""
    calls = source_approximate_qft_calls(width, precision, with_reverse=with_reverse)
    clifford = width + calls.get(("TwoBitSwap", ()), 0)
    toffoli: SymbolicInt = 0
    peak: SymbolicInt = width + precision
    for (name, sizes), count in calls.items():
        if name != "AddIntoPhaseGrad":
            continue
        active, gradient = sizes
        if (
            isinstance(active, int)
            and isinstance(gradient, int)
            and isinstance(width, int)
            and isinstance(precision, int)
        ):
            if active == 0:
                # The source defines a raw negative estimate here although the
                # zero-width addition has no physical action.
                leaf = -2
                leaf_peak = width + precision + 1
            else:
                phase_add = source_phase_add_cost(active, gradient, controlled=True)
                leaf = phase_add.gates.toffoli
                leaf_peak = phase_add.qubits + width + precision - active - gradient - 1
            toffoli += count * leaf
            peak = smax(peak, leaf_peak)
        else:
            toffoli += count * (2 * gradient - 4)
            peak = smax(3, peak, active + gradient + 1)
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=clifford,
        measurement=0,
        toffoli=toffoli,
        logical_qubits=peak,
    )
