# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QvrPhaseGradient and find_optimal_phase_grad_size.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/quantum_variable_rotation.py

"""Concrete precision selection and lowering for phase-gradient variable rotations."""

from dataclasses import dataclass
from math import ceil, isfinite, ldexp, log2, tau
from typing import cast

from sympy import Expr

from ....core.ir import Instruction
from ..._component_trace.resource_model import (
    ComponentCost,
    estimate_component_resources,
)
from ..._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    bit_length,
    smax,
)
from ..._component_trace.symbolic_math import ceil as source_ceil
from ..._component_trace.symbolic_math import log2 as source_log2
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .scaled_phase_addition import (
    _coefficient_word,
    scaled_phase_add_ir,
    source_scaled_phase_add_cost,
)


@dataclass(frozen=True)
class QvrPhaseFormat:
    """Concrete widths selected for the gradient and coefficient."""

    gradient_width: int
    coefficient_width: int
    coefficient_fraction_bits: int


@dataclass(frozen=True)
class QvrPhaseSizing:
    """Separate the source width bound, search result, and selected format."""

    format: QvrPhaseFormat
    formula_width: int
    optimized_width: int | None


def _optimized_width(
    word: int, fraction: int, magnitude: int, gamma_fraction: int, tolerance: float
) -> int:
    """Search the source's worst-case cost word using per-term truncation."""
    cost = ldexp(word, -fraction)
    expected = (magnitude / (1 << gamma_fraction) * cost) % 1
    shifts = tuple(
        bit - gamma_fraction - fraction
        for bit in reversed(range(magnitude.bit_length()))
        if (magnitude >> bit) & 1
    )
    low, high, result = 1, 100, None
    while low <= high:
        mid = (low + high) // 2
        value = 0
        modulus = 1 << mid
        for offset in shifts:
            shift = mid + offset
            # The pinned fixed-point search aligns each term to the
            # accumulator. A negative scale uses floating-point alignment
            # before conversion to the raw integer word.
            if shift < 0:
                value = int(value + word * 2.0**shift) % modulus
            else:
                value = (value + (word << shift)) % modulus
        if abs(value / (1 << mid) - expected) <= tolerance:
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    if result is None:
        raise ValueError("QVR precision search found no gradient width")
    return result


def qvr_phase_sizing(
    cost_width: int, cost_fraction_bits: int, coefficient: float, epsilon: float
) -> QvrPhaseSizing:
    """Report the original concrete bound, search, and fallback."""
    if (
        type(cost_width) is not int
        or type(cost_fraction_bits) is not int
        or cost_fraction_bits > cost_width
        or cost_width < 0
    ):
        raise ValueError("QVR requires a valid unsigned cost format")
    if type(epsilon) not in (int, float) or not 0 < epsilon < tau:
        raise ValueError("QVR epsilon must be positive and smaller than 2π")
    if type(coefficient) not in (int, float) or (
        type(coefficient) is float and not isfinite(coefficient)
    ):
        raise ValueError("QVR requires a finite coefficient")
    ratio = tau / epsilon
    if not isfinite(ratio):
        raise ValueError("QVR epsilon is too small for concrete sizing")
    phase_bits = ceil(log2(ratio))
    gamma_fraction = cost_width - cost_fraction_bits + phase_bits
    gamma_width = int(abs(coefficient)).bit_length() + gamma_fraction
    magnitude = _coefficient_word(coefficient, gamma_width, gamma_fraction)
    terms = min(magnitude.bit_count(), (phase_bits + 2) // 2)
    if not terms:
        raise ValueError(
            "QVR source sizing is undefined for a zero quantized coefficient"
        )
    formula = ceil(log2(terms * tau / epsilon))
    optimized = None
    if cost_width < 54:
        optimized = _optimized_width(
            (1 << cost_width) - 1,
            cost_fraction_bits,
            magnitude,
            gamma_fraction,
            epsilon / tau,
        )
    width = formula if optimized is None else min(formula, optimized)
    return QvrPhaseSizing(
        QvrPhaseFormat(width, gamma_width, gamma_fraction), formula, optimized
    )


def qvr_phase_format(
    cost_width: int, cost_fraction_bits: int, coefficient: float, epsilon: float
) -> QvrPhaseFormat:
    """Select widths with the original concrete precision and fallback rules."""
    return qvr_phase_sizing(cost_width, cost_fraction_bits, coefficient, epsilon).format


def source_symbolic_qvr_gradient_cost(
    cost_width: SymbolicInt,
    cost_fraction_bits: SymbolicInt,
    coefficient: Expr,
    epsilon: SymbolicReal,
) -> SymbolicResourceFormula:
    """Model the source path with a symbolic coefficient, before binding."""
    from sympy import pi as symbolic_pi

    pi = symbolic_pi if isinstance(epsilon, Expr) else tau / 2
    phase = source_ceil(source_log2(2 * pi / epsilon))
    error_terms = (phase + 2) // 2
    width = source_ceil(source_log2(error_terms * 2 * pi / epsilon))
    integer_width = cast(SymbolicInt, smax(0, bit_length(abs(coefficient))))
    coefficient_width = integer_width + cost_width - cost_fraction_bits + phase
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=0,
        measurement=0,
        logical_qubits=cost_width + width,
        toffoli=((coefficient_width + 2) // 2) * (width - 2),
    )


def source_qvr_phase_gradient_cost(
    cost_width: int, cost_fraction_bits: int, coefficient: float, epsilon: float
) -> ComponentCost:
    """Model the locked QVR's scaled-add calls independently of lowering."""
    fmt = qvr_phase_format(cost_width, cost_fraction_bits, coefficient, epsilon)
    scaled = source_scaled_phase_add_cost(
        cost_width,
        cost_fraction_bits,
        fmt.gradient_width,
        coefficient,
        fmt.coefficient_width,
        fmt.coefficient_fraction_bits,
    )
    return ComponentCost(
        "QvrPhaseGradient",
        cost_width + fmt.gradient_width,
        estimate_component_resources(scaled).gates,
    )


def qvr_phase_gradient_ir(
    cost: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Bind the selected format to the source scaled-add graph.

    The caller owns the negative-exponent phase-gradient state and the clean
    work required by the scaled addition. The cost bits stay unchanged.
    """
    fmt = qvr_phase_format(len(cost), cost_fraction_bits, coefficient, epsilon)
    if len(gradient) != fmt.gradient_width:
        raise ValueError(f"QVR requires a {fmt.gradient_width}-bit phase gradient")
    return scaled_phase_add_ir(
        cost,
        gradient,
        workspace,
        classical_bits,
        input_fraction_bits=cost_fraction_bits,
        coefficient=coefficient,
        coefficient_width=fmt.coefficient_width,
        coefficient_fraction_bits=fmt.coefficient_fraction_bits,
        adjoint=adjoint,
    )
