# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SynthesizeLRCircuit at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Stored PLU synthesis with explicit free output aliases and exact inversion."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .field_multiplication import _FieldEmitter
from .field_word import binary_field_word_width


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
    return _linear_circuit_program(width, gates, outputs)


def binary_linear_decomposition_program(
    matrix: tuple[tuple[int, ...], ...], *, adjoint: bool = False
) -> ProgramRecord:
    """Store the literal source recipe, including zero pivots and ignored flag.

    Singular matrices produce reversible circuits that need not equal their
    matrix action. Distinct entry names retain the declared source direction.
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
    outputs = emitter.linear_with_zero_pivots(
        tuple(range(width)), [list(row) for row in matrix]
    )
    return _linear_circuit_program(
        width,
        tuple(emitter.instructions),
        outputs,
        name="linear_adjoint" if adjoint else "main",
    )


def _linear_circuit_program(
    width: int,
    gates: tuple[Instruction, ...],
    outputs: tuple[int, ...],
    *,
    name: str = "main",
) -> ProgramRecord:
    leaf = bit_gate_function(
        "linear_cx",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature((Register("q", quantum_bit_type(), shape=(width,)),)),
        (leaf,),
        name=name,
    )
    bits = np.asarray(builder.inputs["q"], dtype=object).copy()
    for gate in gates:
        control, target = gate.wires
        result = builder.call_dict(
            leaf.name, {"ctrl": bits[control], "target": bits[target]}
        )
        bits[control], bits[target] = result["ctrl"], result["target"]
    return builder.finalize({"q": bits[list(outputs)]})


def field_linear_program(
    dtype: RegisterType,
    matrix: tuple[tuple[int, ...], ...],
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Adapt a binary field word to the shared coefficient-array linear owner."""
    if binary_field_word_width(dtype) != len(matrix):
        raise ValueError("Field linear matrix must match the word width")
    return _field_linear_word_program(
        dtype, binary_linear_program(matrix, adjoint=adjoint)
    )


def field_linear_decomposition_program(
    dtype: RegisterType, matrix: tuple[tuple[int, ...], ...], *, adjoint: bool = False
) -> ProgramRecord:
    """Retain a literal linear child's declared direction inside a field word."""
    if binary_field_word_width(dtype) != len(matrix):
        raise ValueError("Field linear matrix must match the word width")
    return _field_linear_word_program(
        dtype, binary_linear_decomposition_program(matrix, adjoint=adjoint)
    )


def _field_linear_word_program(
    dtype: RegisterType, linear: ProgramRecord
) -> ProgramRecord:
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
    word = builder.inputs["x"]
    assert isinstance(word, RegisterBinding)
    coefficients = builder.split(word)[::-1]
    result = builder.call_program(
        linear, {"q": coefficients}, entry=linear.functions[0].name
    )["q"]
    return builder.finalize(
        {"x": builder.join(np.asarray(result, dtype=object)[::-1], dtype)}
    )
