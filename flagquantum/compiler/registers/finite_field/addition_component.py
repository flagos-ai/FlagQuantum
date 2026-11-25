# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Addition and GF2AddK.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic

"""Typed binary field addition through the existing coefficient emitters."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .coefficients import gf2_add_constant_ir, gf2_add_ir
from .field_word import binary_field_word_width


def field_addition_program(
    dtype: RegisterType, *, constant: int | None = None, adjoint: bool = False
) -> ProgramRecord:
    """XOR a field word or a classical constant into a typed through word.

    The descriptor and polynomial identity already belong to the caller. The
    graph splits each owned word into big-endian bits, applies the existing
    coefficient gate order, and rejoins the original field descriptors.
    Quantum XOR is source-self-adjoint. Constant adjoints reverse their X calls.
    """
    if type(adjoint) is not bool:
        raise ValueError("Field addition direction requires a boolean")
    if dtype.encoding != "finite_field":
        raise ValueError("Field addition requires a binary field descriptor")
    width = binary_field_word_width(dtype)
    if constant is None:
        signature = Signature.build_from_dtypes(x=dtype, y=dtype)
        bit_ports = tuple(Signature.build(ctrl=1, target=1))
        leaf = bit_gate_function(
            "field_cx", bit_ports, CircuitIR(2, (Instruction("cx", (0, 1)),))
        )
        gates = gf2_add_ir(tuple(range(width)), tuple(range(width, 2 * width)))
    else:
        signature = Signature.build_from_dtypes(x=dtype)
        bit_ports = tuple(Signature.build(q=1))
        leaf = bit_gate_function(
            "field_x", bit_ports, CircuitIR(1, (Instruction("x", (0,)),))
        )
        gates = gf2_add_constant_ir(tuple(range(width)), constant)
        if adjoint:
            gates = tuple(reversed(gates))

    builder = ComponentProgramBuilder(signature, (leaf,))
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    lhs = builder.split(x)
    if constant is not None:
        for gate in gates:
            (i,) = gate.wires
            lhs[i] = builder.call_dict(leaf.name, {"q": lhs[i]})["q"]
        return builder.finalize({"x": builder.join(lhs, dtype)})

    y = builder.inputs["y"]
    assert isinstance(y, RegisterBinding)
    rhs = builder.split(y)
    for gate in gates:
        i, other = gate.wires
        j = other - width
        output = builder.call_dict(leaf.name, {"ctrl": lhs[i], "target": rhs[j]})
        lhs[i], rhs[j] = output["ctrl"], output["target"]
    return builder.finalize(
        {"x": builder.join(lhs, dtype), "y": builder.join(rhs, dtype)}
    )
