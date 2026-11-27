# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Partition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py

"""Declare and build a source-ordered shaped Partition or its inverse."""

import numpy as np

from ....core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import classical_bit_type, opaque_register_type
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
    width = ssum(register.total_bits() for register in registers)
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
    """Build quantum wire views or retain a mixed-domain logical declaration.

    Classical parts retain their total storage width, as in the source. They
    cannot physically reinterpret quantum wires as classical slots; the source
    Cirq composite exporter also rejects output classical wires.
    """
    signature = partition_signature(registers, reverse=reverse)
    if any(
        (
            register.dtype == classical_bit_type()
            or register.dtype.parameters[:1] == (("storage", "classical"),)
        )
        and 0 not in register.shape_symbolic
        for register in registers
    ):
        declaration = ProgramRecord(
            (FunctionRecord(name, (), registers=tuple(signature)),)
        )
        diagnostics = verify_program_record(declaration)
        if diagnostics:
            raise ValueError(f"invalid logical Partition declaration: {diagnostics}")
        return declaration
    builder = ComponentProgramBuilder(signature, name=name)
    outputs = rebind_partition(builder, signature[0], registers, reverse=reverse)
    return _namespace_helpers(builder.finalize(outputs), name)


def rebind_partition(
    builder: ComponentProgramBuilder,
    word: Register,
    registers: tuple[Register, ...],
    *,
    reverse: bool,
) -> dict[str, BindingCollection]:
    """Split or join register slices on an existing directional signature."""
    outputs: dict[str, BindingCollection] = {}
    if reverse:
        joined_bits: list[RegisterBinding] = []
        for register in registers:
            for binding in np.asarray(builder.inputs[register.name]).flat:
                joined_bits.extend(builder.split(binding))
        outputs[word.name] = builder.join(joined_bits, word.dtype)
    else:
        packed = builder.inputs[word.name]
        assert isinstance(packed, RegisterBinding)
        bits = builder.split(packed)
        offset = 0
        for register in registers:
            values = np.empty(register.shape, dtype=object)
            n = executable_register_width(register.dtype) if values.size else 0
            for index in register.all_idxs():
                values[index] = builder.join(bits[offset : offset + n], register.dtype)
                offset += n
            outputs[register.name] = values if register.shape else values[()]
    return outputs
