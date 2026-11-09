# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, GF2MulK/GF2ShiftRight/GF2ShiftLeft.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_square.py
# Changes: native IR and explicit polynomial-basis qubits/output mappings.

"""In-place linear maps of binary extension field registers."""

from __future__ import annotations

from typing import cast

from ....core.ir import Instruction
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .field_multiplication import (
    FieldDecomposition,
    _FieldEmitter,
    _product,
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
    free qubit permutation; fixed-qubit consumers can bind it explicitly.
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


def _square(
    emitter: _FieldEmitter,
    qubits: tuple[int, ...],
    count: int = 1,
    *,
    adjoint: bool = False,
) -> tuple[int, ...]:
    """Apply a Frobenius power using the shared PLU emitter."""
    n = len(qubits)
    columns = [1 << i for i in range(n)]
    for _ in range(count % n):
        columns = [_product(value, value, emitter.modulus) for value in columns]
    matrix = [[(columns[j] >> i) & 1 for j in range(n)] for i in range(n)]
    start = len(emitter.instructions)
    result = emitter.linear(qubits[::-1], matrix)[::-1]
    if not adjoint:
        return result
    gates = emitter.instructions[start:]
    del emitter.instructions[start:]
    binding = dict(zip(result, qubits, strict=True))
    emitter.instructions.extend(
        Instruction(g.name, tuple(binding[w] for w in g.wires)) for g in reversed(gates)
    )
    return tuple(binding[w] for w in qubits)


def gf2_frobenius(
    target: tuple[int, ...],
    count: int = 1,
    *,
    modulus: int,
    adjoint: bool = False,
) -> FieldDecomposition:
    """Raise a big-endian GF(2^m) word to 2**count or apply its inverse.

    Return the CNOT stream and the source-style logical output permutation.
    Count zero is the identity. The power is periodic in the field width.
    No clean quantum or classical workspace is required.
    """
    _validate_field(len(target), modulus)
    _validate_registers(target)
    if type(count) is not int or count < 0:
        raise ValueError("Frobenius count must be a nonnegative concrete integer")
    if type(adjoint) is not bool:
        raise ValueError("Frobenius adjoint must be boolean")
    emitter = _FieldEmitter((), modulus)
    output = _square(emitter, target, count, adjoint=adjoint)
    return FieldDecomposition(tuple(emitter.instructions), (output,), ())


def source_gf2_square_resource_bound(width: SymbolicInt) -> SymbolicResourceFormula:
    """Report the locked shape-only PLU bound, separate from the emitted CNOT count."""
    if isinstance(width, (int, float)) and (type(width) is not int or width < 1):
        raise ValueError("Field width must be a positive integer")
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=width * (width - 1),
        measurement=0,
        logical_qubits=cast(SymbolicInt, smax(2, width)),
    )
