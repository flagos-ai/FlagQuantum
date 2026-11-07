# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _Int/_UInt/_MontgomeryUInt.assert_valid_val.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Scalar integer-word value validation for classical component compositions."""

import numpy as np


def validate_integer_word_value(
    value: object, width: int, *, signed: bool = False
) -> None:
    """Check a concrete scalar word without casting or replacing its object.

    NumPy integer scalars follow Python integer bounds; arrays are not scalar
    register values. Unsigned and stored Montgomery words share the full
    width-bounded domain. Source bool values count as Python integers.
    """
    if type(width) is not int or width < 1 or type(signed) is not bool:
        raise ValueError(
            "Integer word validation requires positive width and boolean signedness"
        )
    if not isinstance(value, (int, np.integer)):
        raise ValueError("Scalar integer register requires a Python or NumPy integer")
    low = -(2 ** (width - 1)) if signed else 0
    high = 2 ** (width - 1) if signed else 2**width
    if value < low or value >= high:
        raise ValueError("Classical integer value is outside the register domain")
