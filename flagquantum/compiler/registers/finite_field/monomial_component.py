# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2ShiftLeft/GF2ShiftRight at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Store sequential field shifts and their gate-free physical output aliases."""

from operator import index
from typing import SupportsIndex

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_parameters import is_symbolic_value
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .field_word import field_word_width, raw_field_word_width


def monomial_multiplication_program(
    dtype: RegisterType, power: int, *, modulus: int
) -> ProgramRecord:
    """Multiply MSB field word f by x**power with original sequential shifts.

    Negative powers follow the source left-shift inverse. Each cyclic output
    relabel is free and remains in the returned binding. No work is allocated.
    """
    width = field_word_width(dtype, modulus)
    if type(power) is not int:
        raise ValueError("Monomial exponent must be a concrete integer")
    if not modulus & 1:
        raise ValueError("Monomial multiplication requires a nonzero polynomial root")
    degrees = tuple(i for i in range(1, width) if modulus >> i & 1)
    return _monomial_word_program(dtype, power, degrees, active=width > 1)


def _monomial_word_program(
    dtype: RegisterType, power: int, degrees: tuple[int, ...], *, active: bool
) -> ProgramRecord:
    leaf = bit_gate_function(
        "field_shift_cx",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(f=dtype), (leaf,))
    word = builder.inputs["f"]
    assert isinstance(word, RegisterBinding)
    bits = np.asarray(builder.split(word)[::-1], dtype=object)
    if active:
        for _ in range(abs(power)):
            if power >= 0:
                bits = np.concatenate((bits[-1:], bits[:-1]))
            for i in degrees if power >= 0 else reversed(degrees):
                result = builder.call_dict(
                    leaf.name, {"ctrl": bits[0], "target": bits[i]}
                )
                bits[0], bits[i] = result["ctrl"], result["target"]
            if power < 0:
                bits = np.concatenate((bits[1:], bits[:1]))
        bits = bits[::-1]
    return builder.finalize({"f": builder.join(bits, dtype)})


def monomial_decomposition_program(
    dtype: RegisterType, count: SupportsIndex, *, modulus: int, adjoint: bool = False
) -> ProgramRecord:
    """Preserve source right/left iteration counts, including negative no-ops.

    adjoint selects the source left-shift constructor. Its negative count still
    runs no iterations, unlike a negative mathematical exponent.
    """
    if is_symbolic_value(count) or not isinstance(count, SupportsIndex):
        raise ValueError("Literal field shift count requires a concrete integer")
    if type(adjoint) is not bool:
        raise ValueError("Literal field shift direction requires a boolean")
    raw_field_word_width(dtype)
    metadata = dict(dtype.parameters)
    characteristic, degree = metadata["characteristic"], metadata["degree"]
    if type(characteristic) is not int or type(degree) is not int:
        raise ValueError("Literal field shifts require concrete field metadata")
    if type(modulus) is not int or modulus <= 0:
        raise ValueError("Literal field shift polynomial must be a positive integer")
    if metadata.get("polynomial", modulus) != modulus:
        raise ValueError("Field modulus must match the field descriptor")
    value, position = modulus, 0
    nonzero: list[int] = []
    while value:
        value, coefficient = divmod(value, characteristic)
        if coefficient:
            nonzero.append(position)
        position += 1
    if not nonzero or nonzero[-1] != degree:
        raise ValueError("Literal shift polynomial must match the field degree")
    steps = max(index(count), 0)
    return _monomial_word_program(
        dtype, -steps if adjoint else steps, tuple(nonzero[1:-1]), active=degree > 1
    )
