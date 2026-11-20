# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2ShiftLeft/GF2ShiftRight at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Store sequential field shifts and their gate-free physical output aliases."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .field_word import field_word_width


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
    leaf = bit_gate_function(
        "field_shift_cx",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(f=dtype), (leaf,))
    word = builder.inputs["f"]
    assert isinstance(word, RegisterBinding)
    bits = np.asarray(builder.split(word)[::-1], dtype=object)
    degrees = tuple(i for i in range(1, width) if modulus >> i & 1)
    if width > 1:
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
    return builder.finalize({"f": builder.join(bits[::-1], dtype)})
