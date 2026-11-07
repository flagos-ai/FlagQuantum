# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Square.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py

"""Classical square production and checked erasure on the source valid subspace."""

from collections.abc import Mapping

from .classical_bitwise import IntegerValue


def square_classical_values(
    values: Mapping[str, IntegerValue], *, uncompute: bool = False
) -> dict[str, IntegerValue]:
    """Square ``a``, or consume ``result`` only when it equals ``a**2``.

    Forward evaluation ignores an extra result, matching the original named
    value method. Erasure preserves the original equality assertion, including
    NumPy ambiguous-truth failures. Missing fields raise KeyError. No width
    reduction or input casts are added; typed validation is a separate stage.
    """
    if uncompute:
        value, result = values["a"], values["result"]
        if not result == value**2:
            raise AssertionError
        return {"a": value}
    value = values["a"]
    return {"a": value, "result": value**2}
