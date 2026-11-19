# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BiQubitsMixer and SingleQubitCompare.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Own directional comparison flags over the existing exact physical recipes."""

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .comparison_tree import bi_qubits_mixer_ir, single_qubit_compare_ir


def mixer_program(*, adjoint: bool = False) -> ProgramRecord:
    """Mix two two-bit words, producing or consuming three owned flag qubits.

    The forward flags start clean. The exact adjoint requires the forward state,
    erases its two And flags by measurement and frees all three work qubits.
    """
    gates = bi_qubits_mixer_ir(
        (0, 1),
        (2, 3),
        (4, 5, 6),
        adjoint=adjoint,
        classical_bits=(0, 1) if adjoint else (),
    )
    bit, word, work = (
        quantum_bit_type(),
        unsigned_register_type(2),
        RegisterType("opaque", 3),
    )
    leaf = bit_gate_function(
        "mixer_gates", (Register("bits", bit, (7,)),), CircuitIR(7, gates)
    )
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", word),
                Register("y", word),
                Register("ancilla", work, side=side),
            )
        ),
        (leaf,),
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    ancilla = builder.inputs["ancilla"] if adjoint else builder.allocate(dtype=work)
    assert isinstance(ancilla, RegisterBinding)
    bits = np.asarray(
        builder.call_dict(
            "mixer_gates",
            {
                "bits": np.array(
                    [*builder.split(x), *builder.split(y), *builder.split(ancilla)],
                    dtype=object,
                )
            },
        )["bits"]
    )
    outputs = {"x": builder.join(bits[:2], word), "y": builder.join(bits[2:4], word)}
    ancilla = builder.join(bits[4:], work)
    if adjoint:
        builder.free(ancilla)
    else:
        outputs["ancilla"] = ancilla
    return builder.finalize(outputs)


def single_qubit_comparison_program(*, adjoint: bool = False) -> ProgramRecord:
    """Produce comparison flags and equality-in-b, or consume their exact image."""
    gates = single_qubit_compare_ir(
        0, 1, 2, 3, adjoint=adjoint, classical_bits=(0,) if adjoint else ()
    )
    bit = quantum_bit_type()
    leaf = bit_gate_function(
        "compare_gates", (Register("bits", bit, (4,)),), CircuitIR(4, gates)
    )
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("a", bit),
                Register("b", bit),
                Register("less_than", bit, side=side),
                Register("greater_than", bit, side=side),
            )
        ),
        (leaf,),
    )
    less = builder.inputs["less_than"] if adjoint else builder.allocate(dtype=bit)
    greater = builder.inputs["greater_than"] if adjoint else builder.allocate(dtype=bit)
    result = np.asarray(
        builder.call_dict(
            "compare_gates",
            {
                "bits": np.array(
                    [builder.inputs["a"], builder.inputs["b"], less, greater],
                    dtype=object,
                )
            },
        )["bits"]
    )
    outputs = {"a": result[0], "b": result[1]}
    if adjoint:
        builder.free(result[2])
        builder.free(result[3])
    else:
        outputs.update(less_than=result[2], greater_than=result[3])
    return builder.finalize(outputs)
