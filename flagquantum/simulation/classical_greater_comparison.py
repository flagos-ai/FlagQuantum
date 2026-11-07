# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearDepthGreaterThan and CLinearDepthGreaterThan.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Linear comparison value actions with distinct target and activation policies."""

from typing import cast

from .classical_bitwise import IntegerValue


def linear_greater_classical_values(
    left: IntegerValue, right: IntegerValue, target: IntegerValue
) -> tuple[IntegerValue, IntegerValue, IntegerValue]:
    """Keep the source binary-target assertion and conditional add/modulo update.

    The source numerical method ignores its signed flag and width. Values are
    compared as supplied; quantum signed difference-sign discrepancies remain.
    """
    assert target == target % 2
    if left > right:
        target = (target + 1) % 2
    return left, right, target


def controlled_linear_greater_classical_values(
    control: IntegerValue,
    left: IntegerValue,
    right: IntegerValue,
    target: IntegerValue,
    *,
    control_value: int = 1,
) -> tuple[IntegerValue, IntegerValue, IntegerValue, IntegerValue]:
    """Use exact activation equality, preserving inactive objects before comparison.

    Unlike the unconditional comparator this method adds no binary-target check.
    The source numerical constructor does not restrict cv to zero or one; the
    compiler's stricter executable control requirements remain separate.
    """
    if control == control_value:
        return control, left, right, cast(IntegerValue, target ^ (left > right))
    return control, left, right, target
