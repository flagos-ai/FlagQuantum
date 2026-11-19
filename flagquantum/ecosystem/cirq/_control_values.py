# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpec.to_cirq_cv/from_cirq_cv.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Translate one quantum equality clause at the optional Cirq boundary."""

from math import prod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.value_parameters import is_symbolic_value
from ...simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_value,
)
from ...simulation.classical_typed_values import collect_typed_register_value
from ._register_export import _register_shape
from .conversion import _cirq


def export_control_values(
    registers: tuple[Register, ...], values: tuple[NDArray[Any], ...]
) -> Any:
    """Export row-major, MSB-first equalities as one Cirq SumOfProducts clause.

    Registers retain their native metadata. Classical and symbolic declarations
    cannot be represented by this quantum-only SDK boundary. Scalar codec
    domain errors propagate before a foreign control object is returned.
    """
    if len(registers) != len(values):
        raise ValueError("control registers require one activation array each")
    if is_symbolic_value(*registers, *values):
        raise ValueError("cannot export symbolic control values to Cirq")
    bits: list[int] = []
    for register, array in zip(registers, values, strict=True):
        _register_shape(register)
        if array.shape != register.shape:
            raise ValueError("control activation shape differs from its register")
        for index in register.all_idxs():
            bits.extend(encode_register_value(register, array[index]))
    return _cirq().SumOfProducts([tuple(bits)])


def import_control_values(
    control_values: Any, *, registers: tuple[Register, ...] | None = None
) -> tuple[tuple[Register, ...], tuple[NDArray[Any], ...]]:
    """Decode one Cirq AND clause into native metadata and activation arrays.

    Omitted registers produce a scalar QBit for one control, or one flat QBit
    array otherwise. Explicit register objects are retained. Unlike the frozen
    source's multi-register cursor defect, every register consumes its own slice.
    Multi-clause expressions, nonbinary controls and incompatible widths fail.
    """
    clauses = tuple(control_values.expand())
    if len(clauses) != 1:
        raise ValueError("control values require exactly one AND clause")
    bits = clauses[0]
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("control values require binary quantum operands")
    if registers is None:
        shape = (len(bits),) if len(bits) > 1 else ()
        registers = (Register("ctrl", quantum_bit_type(), shape),)
    layouts = tuple(_register_shape(register) for register in registers)
    if sum(prod(layout) for layout in layouts) != len(bits):
        raise ValueError("control bit count differs from the declared registers")
    offset = 0
    values: list[NDArray[Any]] = []
    for register, layout in zip(registers, layouts, strict=True):
        end = offset + prod(layout)
        words = np.asarray(bits[offset:end]).reshape(layout)
        scalar = Register(register.name, register.dtype, side=register.side)
        decoded = {
            index: decode_register_bits(scalar, words[index])
            for index in register.all_idxs()
        }
        values.append(np.asanyarray(collect_typed_register_value(register, decoded)))
        offset = end
    return registers, tuple(values)
