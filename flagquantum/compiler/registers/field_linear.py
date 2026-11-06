# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, GF2MulK/GF2ShiftRight/GF2ShiftLeft.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# Changes: native IR and explicit polynomial-basis wires/output mappings.

"""In-place binary-field constant and signed monomial multiplication."""

from .field_multiplication import (
    FieldDecomposition,
    _FieldEmitter,
    _validate_field,
)
from .polynomial import _validate_registers


def gf2_constant_multiplication(
    target: tuple[int, ...], value: int, *, modulus: int
) -> FieldDecomposition:
    """Multiply a big-endian field register by a nonzero concrete constant.

    The PLU CNOT network and logical output relabeling follow GF2MulK exactly.
    A zero constant is not an invertible in-place operation and is rejected.
    No clean quantum or classical workspace is required.
    """
    _validate_field(len(target), modulus)
    _validate_registers(target)
    if type(value) is not int or not 0 < value < 1 << len(target):
        raise ValueError(
            "Field multiplier must be a nonzero integer fitting the register"
        )
    emitter = _FieldEmitter((), modulus)
    result = emitter.constant(target[::-1], value)[::-1]
    return FieldDecomposition(tuple(emitter.instructions), (result,), ())


def gf2_monomial_multiplication(
    target: tuple[int, ...], power: int, *, modulus: int
) -> FieldDecomposition:
    """Multiply by x**power modulo an explicit irreducible polynomial.

    Positive powers port GF2ShiftRight; negative powers port GF2ShiftLeft.
    Gate ordering is preserved rather than reducing the exponent modulo the
    field order. Compilation scales with abs(power). Outputs retain the source's
    free wire permutation; fixed-wire consumers can bind it explicitly.
    """
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(target)
    if type(power) is not int:
        raise ValueError("Monomial exponent must be a concrete integer")
    if not modulus & 1:
        raise ValueError("Monomial multiplication requires a nonzero polynomial root")
    emitter = _FieldEmitter((), modulus)
    result = target[::-1]
    if power >= 0:
        result = emitter.shift(result, power)
    elif width > 1:
        degrees = [i for i in range(1, width) if modulus >> i & 1]
        for _ in range(-power):
            for i in reversed(degrees):
                emitter.cx(result[0], result[i])
            result = result[1:] + result[:1]
    return FieldDecomposition(tuple(emitter.instructions), (result[::-1],), ())
