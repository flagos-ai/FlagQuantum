# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Partition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py

"""Declare and build a source-ordered shaped Partition or its inverse."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_math import ssum
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _namespace_helpers


def partition_signature(
    registers: tuple[Register, ...], *, reverse: bool = False
) -> Signature:
    """Keep the packed word first and the shaped words in declaration order."""
    width = ssum(register.total_qubits() for register in registers)
    word = Register("x", opaque_register_type(width), side=RegisterSide.LEFT)
    parts = tuple(
        Register(
            register.name, register.dtype, register.shape_symbolic, RegisterSide.RIGHT
        )
        for register in registers
    )
    signature = Signature((word, *parts))
    return signature.adjoint() if reverse else signature


def partition_component_program(
    registers: tuple[Register, ...], name: str, reverse: bool
) -> ProgramRecord:
    """Build a concrete partition or its inverse using existing split/join calls."""
    signature = partition_signature(registers, reverse=reverse)
    word = signature[0]
    builder = ComponentProgramBuilder(signature, name=name)
    outputs: dict[str, BindingCollection] = {}
    if reverse:
        joined_bits: list[RegisterBinding] = []
        for register in registers:
            for binding in np.asarray(builder.inputs[register.name]).flat:
                joined_bits.extend(builder.split(binding))
        outputs["x"] = builder.join(joined_bits, word.dtype)
    else:
        packed = builder.inputs["x"]
        assert isinstance(packed, RegisterBinding)
        bits = builder.split(packed)
        offset = 0
        for register in registers:
            values = np.empty(register.shape, dtype=object)
            n = executable_register_width(register.dtype)
            for index in register.all_idxs():
                values[index] = builder.join(bits[offset : offset + n], register.dtype)
                offset += n
            outputs[register.name] = values if register.shape else values[()]
    return _namespace_helpers(builder.finalize(outputs), name)
