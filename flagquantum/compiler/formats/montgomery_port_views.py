# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Montgomery port metadata follows Qualtran v0.7.0 serialization/data_types.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/data_types.py

"""Bridge the source codec's optional-modulus omission with existing Casts."""

from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import montgomery_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.layout.component_cast import cast_word_function


def source_modulus_omission(outer: Register, inner: Register) -> bool:
    """Recognize only the known modulus omission, preserving every other field."""
    return (
        outer.name == inner.name
        and outer.side == inner.side
        and outer.shape_symbolic == inner.shape_symbolic
        and inner.dtype.encoding == "montgomery"
        and tuple(name for name, _ in inner.dtype.parameters) == ("modulus",)
        and outer.dtype == montgomery_register_type(inner.dtype.num_qubits)
    )


def restore_montgomery_ports(
    program: ProgramRecord, registers: tuple[Register, ...]
) -> ProgramRecord:
    """Bridge only validated source modulus omissions with existing typed views."""
    native = program.functions[0].registers
    assert native is not None
    builder = ComponentProgramBuilder(Signature(registers))
    arguments = dict(builder.inputs)
    for outer, inner in zip(registers, native, strict=True):
        if outer.side & RegisterSide.LEFT and outer.dtype != inner.dtype:
            name = f"modulus_input_{outer.name}"
            builder.define(
                cast_word_function(name, outer.dtype, inner.dtype, outer.shape)
            )
            arguments[outer.name] = builder.call_dict(
                name, {"reg": arguments[outer.name]}
            )["reg"]
    outputs = builder.call_program(program, arguments)
    for outer, inner in zip(registers, native, strict=True):
        if outer.side & RegisterSide.RIGHT and outer.dtype != inner.dtype:
            name = f"modulus_output_{outer.name}"
            builder.define(
                cast_word_function(name, inner.dtype, outer.dtype, outer.shape)
            )
            outputs[outer.name] = builder.call_dict(name, {"reg": outputs[outer.name]})[
                "reg"
            ]
    return builder.finalize(outputs)
