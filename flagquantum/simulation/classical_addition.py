# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 simulation.classical_sim.add_ints and arithmetic
# Add, AddK and OutOfPlaceAdder.on_classical_vals at the pinned revision.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Classical integer addition models, independent of quantum instruction emission."""

from typing import TYPE_CHECKING, cast

from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Expr


def add_integer_values(
    left: IntegerValue,
    right: IntegerValue,
    *,
    num_bits: int | None = None,
    signed: bool = False,
) -> IntegerValue:
    """Add with the source's optional finite-width wrap and NumPy promotion.

    Without width, use arbitrary-precision integer addition. With nonnegative
    width, wrap into the unsigned interval or its centered signed interval.
    Do not cast operands: consumers own the original conversion policy.
    """
    result = left + right
    if num_bits is not None:
        modulus = 2**num_bits
        if signed:
            return cast(IntegerValue, (result + modulus // 2) % modulus - modulus // 2)
        return cast(IntegerValue, result % modulus)
    return result


def register_add_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: "int | Expr",
    *,
    signed: bool = False,
) -> tuple[IntegerValue, int]:
    """Preserve left and add its integer value into a target of the given width.

    Original Add casts both values and the target width to Python integers.
    Symbolic integer literals can therefore evaluate here, while unresolved
    symbols retain the original integer-conversion failure.
    """
    result = add_integer_values(
        int(left), int(right), num_bits=int(width), signed=signed
    )
    return left, cast(int, result)


def add_constant_classical_value(
    target: IntegerValue,
    constant: "int | Expr",
    width: "int | Expr",
    *,
    signed: bool = False,
) -> int:
    """Evaluate original AddK conversions, rejecting symbolic width or constant."""
    if not isinstance(constant, int) or not isinstance(width, int):
        from sympy import Expr

        if isinstance(constant, Expr) or isinstance(width, Expr):
            raise ValueError(
                "Classical constant addition does not support symbolic parameters"
            )
    return cast(
        int,
        add_integer_values(
            int(target), int(constant), num_bits=cast(int, width), signed=signed
        ),
    )


def out_of_place_sum_classical_values(
    left: IntegerValue,
    right: IntegerValue,
    width: "int | Expr",
    *,
    include_msb: bool = True,
    adjoint: bool = False,
    output: IntegerValue | None = None,
) -> dict[str, IntegerValue]:
    """Create a sum value or remove the supplied logical output as upstream does.

    Original classical adjoint checks only that output is present, not that it
    equals the sum. It is not a certificate of quantum erasure on wrong outputs.
    The source's normal-mode assertion failures remain explicit exceptions.
    """
    if not isinstance(width, int):
        from sympy import Expr

        if isinstance(width, Expr):
            raise ValueError(
                "Classical out-of-place addition does not support symbolic width"
            )
    if adjoint:
        if output is None:
            raise AssertionError("Classical sum adjoint requires its output value")
        return {"a": left, "b": right}
    if output is not None:
        raise AssertionError("Classical forward sum allocates its own output")
    return {
        "a": left,
        "b": right,
        "c": add_integer_values(
            int(left), int(right), num_bits=cast(int, width) + int(include_msb)
        ),
    }
