# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Compose the source complement/constant shell around modular accumulation."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .controlled_modular_addition_component import controlled_modular_addition_program


def controlled_modular_subtraction_program(
    dtype: RegisterType, modulus: int, *, control_value: int = 1
) -> ProgramRecord:
    """Preserve the source input shell and subtract x from y modulo modulus.

    Words are MSB first and retain uint or Montgomery descriptors. Inputs
    promise residues. The forward and exact inverse constant loaders retain
    the same modulus+1. The modular child owns its clean arithmetic work.
    """
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular subtraction requires unsigned or Montgomery words")
    if type(modulus) is not int or not 0 < modulus < (1 << width) - 1:
        raise ValueError("Modular subtraction requires modulus+1 to fit the operand")
    load = constant_addition_program(dtype, modulus + 1)
    unload = constant_addition_program(dtype, modulus + 1, adjoint=True)
    addition = controlled_modular_addition_program(
        dtype, modulus, control_value=control_value
    )
    return controlled_modular_subtraction_component_program(
        dtype, load, unload, addition
    )


def controlled_modular_subtraction_component_program(
    dtype: RegisterType,
    load: ProgramRecord,
    unload: ProgramRecord,
    addition: ProgramRecord,
) -> ProgramRecord:
    """Preserve the five-call source shell around its explicit child owners.

    Load/unload keep their constructor's constant and exact direction. The
    addition child retains the source residue-domain and cleanup promises;
    unresolved source leaves remain nonexecutable declarations.
    """
    executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular subtraction requires unsigned or Montgomery words")
    complement = bitwise_word_program(dtype, "bitwise_not")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=dtype, y=dtype)
    )
    x = builder.call_program(complement, {"x": builder.inputs["x"]})["x"]
    x = builder.call_program(load, {"x": x}, entry=load.functions[0].name)["x"]
    added = builder.call_program(
        addition, {"ctrl": builder.inputs["ctrl"], "x": x, "y": builder.inputs["y"]}
    )
    x = builder.call_program(unload, {"x": added["x"]}, entry=unload.functions[0].name)[
        "x"
    ]
    x = builder.call_program(complement, {"x": x})["x"]
    return builder.finalize({"ctrl": added["ctrl"], "x": x, "y": added["y"]})
