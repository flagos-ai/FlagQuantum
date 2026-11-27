# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QvrZPow.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/quantum_variable_rotation.py

"""Direct variable rotation of a fixed-point cost register."""

from math import ceil, isfinite, ldexp, log2, pi, tau
from typing import cast

from ....core.ir import Instruction
from ..._component_trace.resource_classification import classify_resource_gate
from ..._component_trace.resource_generalization import RotationCall
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smin
from ..._component_trace.symbolic_math import ceil as source_ceil
from ..._component_trace.symbolic_math import log2 as source_log2
from ..._component_trace.terminal_t_cost import rotation_t_cost
from ..bits.bitwise import _validate_wires


def _fractional_rotations(
    fraction_bits: SymbolicInt, coefficient: SymbolicReal, epsilon: SymbolicReal
) -> SymbolicInt:
    if isinstance(coefficient, (int, float)) and isinstance(epsilon, (int, float)):
        if not isfinite(coefficient) or coefficient == 0:
            raise ValueError("QVR Z power requires a finite nonzero coefficient")
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("QVR Z power requires a finite positive epsilon")
        ratio = tau * abs(coefficient) / epsilon
        if not isfinite(ratio) or ratio == 0:
            raise ValueError(
                "QVR Z power coefficient and epsilon ratio must be finite and positive"
            )
        bound = ceil(log2(ratio))
    else:
        from sympy import pi as symbolic_pi

        two_pi = tau if isinstance(epsilon, (float, int)) else 2 * symbolic_pi
        bound = source_ceil(source_log2(two_pi * abs(coefficient) / epsilon))
    return cast(SymbolicInt, smin(fraction_bits, bound))


def source_qvr_zpow_declaration(
    cost_width: SymbolicInt,
    cost_fraction_bits: SymbolicInt,
    coefficient: SymbolicReal,
    epsilon: SymbolicReal,
    *,
    signed: bool = False,
) -> tuple[RotationCall, SymbolicInt]:
    """Keep the source declared exponent and multiplicity, including negative counts."""
    fractional = _fractional_rotations(cost_fraction_bits, coefficient, epsilon)
    count = cost_width - cost_fraction_bits + fractional + int(signed)
    if count == 0:
        raise ValueError("QVR Z power source precision allocation has zero rotations")
    return RotationCall("zpow", coefficient / 2**fractional, epsilon / count), count


def source_qvr_zpow_cost(
    cost_width: int, cost_fraction_bits: int, coefficient: float, epsilon: float
) -> ComponentCost:
    call, count = source_qvr_zpow_declaration(
        cost_width, cost_fraction_bits, coefficient, epsilon
    )
    assert isinstance(count, int)
    family = classify_resource_gate(call)
    return ComponentCost(
        "QvrZPow",
        cost_width,
        GateCost(
            rotation=count * (family == "rotation"),
            t=count * (family == "t"),
            clifford=count * (family == "clifford"),
        ),
    )


def source_qvr_zpow_synthesis_t_count(
    cost_width: SymbolicInt,
    cost_fraction_bits: SymbolicInt,
    coefficient: SymbolicReal,
    epsilon: SymbolicReal,
) -> SymbolicInt:
    """Evaluate the tutorial's per-declared-rotation synthesis estimate."""
    call, count = source_qvr_zpow_declaration(
        cost_width, cost_fraction_bits, coefficient, epsilon
    )
    return rotation_t_cost(call.precision) * count


def qvr_zpow_schedule(
    cost: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    signed: bool = False,
    adjoint: bool = False,
) -> tuple[tuple[int, float, float], ...]:
    """Return source (qubit, Z exponent, synthesis tolerance) in gate order.

    The concrete unsigned route visits retained qubits from least to most
    significant. The inverse reverses the same schedule and negates its
    exponents. A negative source rotation count skips the rotation loop,
    retaining the standalone sign phase in a signed layout. A signed
    layout with a positive rotation count exceeds the source register.
    """
    _validate_wires(cost)
    width = len(cost)
    if type(cost_fraction_bits) is not int or cost_fraction_bits > width:
        raise ValueError("QVR Z power requires a valid cost fraction width")
    if type(signed) is not bool or type(adjoint) is not bool:
        raise ValueError("QVR Z power mode flags must be boolean")
    if signed and (width < 2 or cost_fraction_bits == width):
        raise ValueError("QVR Z power requires a valid signed cost format")
    if type(coefficient) not in (int, float):
        raise ValueError("QVR Z power requires a finite nonzero coefficient")
    if type(epsilon) not in (int, float):
        raise ValueError("QVR Z power requires a finite positive epsilon")

    fractional = cast(
        int, _fractional_rotations(cost_fraction_bits, coefficient, epsilon)
    )
    rotations = width - cost_fraction_bits + fractional + signed
    if rotations == 0:
        raise ValueError("QVR Z power source precision allocation has zero rotations")
    if signed and rotations > 0:
        raise ValueError(
            "QVR Z power source rotation indexes beyond the signed cost register"
        )
    tolerance = epsilon / rotations
    sign_phase = ((cost[0], -1.0 if adjoint else 1.0, tolerance),) if signed else ()
    offset = 1 + cost_fraction_bits - fractional
    scale = -coefficient if adjoint else coefficient
    order = reversed(range(rotations)) if adjoint else range(rotations)
    return sign_phase + tuple(
        (cost[-(i + offset)], ldexp(scale, i - fractional + 1), tolerance)
        for i in order
    )


def qvr_zpow_ir(
    cost: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    signed: bool = False,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit native phase instructions without synthesizing the rotations."""
    return tuple(
        Instruction("phase", (wire,), params={"theta": pi * exponent})
        for wire, exponent, _ in qvr_zpow_schedule(
            cost,
            cost_fraction_bits=cost_fraction_bits,
            coefficient=coefficient,
            epsilon=epsilon,
            signed=signed,
            adjoint=adjoint,
        )
    )


def symbolic_qvr_zpow_schedule(
    cost: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: SymbolicReal,
    epsilon: SymbolicReal,
    signed: bool = False,
) -> tuple[tuple[int, SymbolicReal, SymbolicReal], ...]:
    """Retain the original expression dispatch when rotation indices are finite.

    The source uses the entire cost width if its rotation count is a SymPy
    expression, even a constant. Do not replace this branch with the numerical
    schedule. Unresolved or out-of-range indices have no finite source body.
    """
    from sympy import Integer

    _validate_wires(cost)
    width = len(cost)
    if type(cost_fraction_bits) is not int or cost_fraction_bits > width:
        raise ValueError("QVR Z power requires a valid cost fraction width")
    if (
        type(signed) is not bool
        or signed
        and (width < 2 or cost_fraction_bits == width)
    ):
        raise ValueError("QVR Z power requires a valid signed cost format")
    fractional = _fractional_rotations(cost_fraction_bits, coefficient, epsilon)
    declared = width - cost_fraction_bits + fractional + int(signed)
    rotations = declared if isinstance(declared, int) else width
    offset = 1 + cost_fraction_bits - fractional
    if not isinstance(offset, (int, Integer)):
        raise ValueError("QVR Z power source rotation indices remain symbolic")
    if rotations == 0:
        raise ValueError("QVR Z power source precision allocation has zero rotations")
    tolerance = epsilon / rotations
    phases: list[tuple[int, SymbolicReal, SymbolicReal]] = (
        [(cost[0], 1, tolerance)] if signed else []
    )
    try:
        for i in range(rotations):
            phases.append(
                (
                    cost[-(i + int(offset))],
                    (2 ** (i - fractional)) * coefficient * 2,
                    tolerance,
                )
            )
    except IndexError as error:
        raise ValueError(
            "QVR Z power source rotation index exceeds cost word"
        ) from error
    return tuple(phases)
