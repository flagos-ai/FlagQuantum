# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GreaterThan.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR and explicit shared tree comparator workspace.

"""Unsigned greater-than as the source complement of less-or-equal."""

from ....core.ir import Instruction
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .less_equal import less_equal_ir, less_equal_requirements


def greater_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR a>b into an arbitrary target, retaining the source tree and final X.

    Big-endian inputs may differ in width. Work sizing/cleanup is owned by the
    shared less_equal_ir and its less_equal_requirements query.
    """
    return (
        *less_equal_ir(left, right, target, workspace, classical_bits),
        Instruction("x", (target,)),
    )


def greater_adjoint_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR a>b using the original X / self-adjoint LessThanEqual inverse.

    Big-endian unsigned operands may differ in width; target may be arbitrary.
    Keep original component order and the shared clean-work/slot requirements.
    Quantum work starts/ends zero; classical slots retain latest outcomes.
    """
    comparison = less_equal_ir(left, right, target, workspace, classical_bits)
    return (Instruction("x", (target,)), *comparison)


def source_greater_cost(left_width: int, right_width: int) -> ComponentCost:
    """Model the source LessThanEqual tree followed by the final X."""
    work, measurements = less_equal_requirements(left_width, right_width)
    ports = left_width + right_width + 1
    overlap = min(left_width, right_width)
    excess = abs(left_width - right_width)
    cliffords = 22 * overlap - 9  # Mixer/adjoint pairs and final one-bit compare.
    if excess:
        # The equal-width final X/CX and the unequal-width final And pair/CX
        # each contribute two Clifford gates. Count the extra prefix separately.
        if excess <= 2:
            cliffords += 4 if excess == 1 else 1
        else:
            cliffords += 5 * excess - 1
        cliffords += 2 * (right_width > left_width)
    return ComponentCost(
        "GreaterThan",
        ports,
        GateCost(clifford=1),
        (
            CostCall(
                ComponentCost(
                    "LessThanEqual",
                    ports + work,
                    GateCost(
                        and_bloq=measurements,
                        clifford=cliffords,
                        measurement=measurements,
                    ),
                )
            ),
        ),
    )
