# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SynthesizeLRCircuit at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Stored PLU synthesis with explicit free output aliases and exact inversion."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .field_multiplication import _FieldEmitter


def binary_linear_program(
    matrix: tuple[tuple[int, ...], ...], *, adjoint: bool = False
) -> ProgramRecord:
    """Apply an invertible binary matrix to a shaped bit array q.

    Array index is the matrix coefficient index. Output aliases implement the
    source's free PLU permutation. The inverse reverses the emitted network and
    its aliases exactly. Both directions require zero work or measurement slots.
    """
    width = len(matrix)
    if not width or any(
        len(row) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in row)
        for row in matrix
    ):
        raise ValueError("Linear synthesis requires a nonempty square binary matrix")
    if type(adjoint) is not bool:
        raise ValueError("Linear synthesis direction requires a boolean")
    emitter = _FieldEmitter((), 0)
    try:
        outputs = emitter.linear(tuple(range(width)), [list(row) for row in matrix])
    except StopIteration:
        raise ValueError("Linear synthesis requires an invertible matrix") from None
    gates = tuple(emitter.instructions)
    if adjoint:
        binding = dict(zip(outputs, range(width), strict=True))
        gates = tuple(
            Instruction(g.name, tuple(binding[w] for w in g.wires))
            for g in reversed(gates)
        )
        outputs = tuple(binding[w] for w in range(width))
    leaf = bit_gate_function(
        "linear_cx",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature((Register("q", quantum_bit_type(), shape=(width,)),)), (leaf,)
    )
    bits = np.asarray(builder.inputs["q"], dtype=object).copy()
    for gate in gates:
        control, target = gate.wires
        result = builder.call_dict(
            leaf.name, {"ctrl": bits[control], "target": bits[target]}
        )
        bits[control], bits[target] = result["ctrl"], result["target"]
    return builder.finalize({"q": bits[list(outputs)]})
