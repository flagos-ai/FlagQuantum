# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpecAnd and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py

"""Give the existing predicate emitter directional typed ports and lifetimes."""

from math import prod

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from .control_predicate import ControlRegister
from .multi_control import multi_and_ir


def validate_control_ports(
    controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
) -> int:
    """Validate finite integer clauses and return their physical bit count."""
    if not controls or len(controls) != len(patterns):
        raise ValueError("control ports require one activation pattern per register")
    Signature(controls)
    for register, pattern in zip(controls, patterns, strict=True):
        encoding = {
            "bit": "unsigned",
            "opaque": "unsigned",
            "uint": "unsigned",
        }.get(register.dtype.encoding)
        if (
            register.side != RegisterSide.THRU
            or register.dtype.classical_bits != 0
            or encoding != pattern.encoding
            or executable_register_width(register.dtype) != pattern.width
            or prod(register.shape) != len(pattern.values)
        ):
            raise ValueError("control port dtype, shape and activation must agree")
        pattern.bits()
    count = sum(len(pattern.bits()) for pattern in patterns)
    if not count:
        raise ValueError("structured control requires a nonempty activation")
    return count


def control_predicate_program(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...],
    *,
    name: str = "predicate",
    uncompute: bool = False,
) -> ProgramRecord:
    """Produce or consume target/junk using row-major MSB-first control words.

    Forward owns clean allocation of n-1 prefix outputs. Erasure requires their
    correct computed values, measures in reverse ladder order and releases them
    after clearing. These logical outputs remain live across an intervening
    controlled call. The current Program admission permits unparameterized bit,
    opaque and unsigned quantum words. Other descriptors raise ValueError.
    """
    count = validate_control_ports(controls, patterns)
    if count < 2 or type(uncompute) is not bool:
        raise ValueError("predicate requires at least two bits and a boolean direction")
    if {r.name for r in controls} & {"junk", "target"}:
        raise ValueError("predicate control names conflict with its logical outputs")
    side = RegisterSide.LEFT if uncompute else RegisterSide.RIGHT
    junk = (
        (Register("junk", RegisterType("opaque", count - 2), side=side),)
        if count > 2
        else ()
    )
    registers = (*controls, *junk, Register("target", quantum_bit_type(), side=side))
    gates = multi_and_ir(
        tuple(range(count)),
        tuple(bit for pattern in patterns for bit in pattern.bits()),
        tuple(range(count, 2 * count - 1)),
        classical_bits=tuple(range(count - 1)) if uncompute else (),
        uncompute=uncompute,
    )
    leaf = bit_gate_function(
        "gates",
        (Register("bits", quantum_bit_type(), (2 * count - 1,)),),
        CircuitIR(2 * count - 1, gates),
    )
    builder = ComponentProgramBuilder(Signature(registers), (leaf,), name=name)
    flat = [
        bit
        for register in controls
        for word in np.asarray(builder.inputs[register.name]).flat
        for bit in builder.split(word)
    ]
    if uncompute:
        junk_input = builder.inputs.get("junk")
        if junk:
            assert isinstance(junk_input, RegisterBinding)
            work = list(builder.split(junk_input))
        else:
            work = []
        work.append(builder.inputs["target"])
    else:
        work = list(builder.split(builder.allocate(count - 1)))
    returned = np.asarray(
        builder.call_dict("gates", {"bits": np.array([*flat, *work], dtype=object)})[
            "bits"
        ]
    )
    outputs: dict[str, BindingCollection] = {}
    cursor = 0
    for register in controls:
        array = np.empty(register.shape, dtype=object)
        width = executable_register_width(register.dtype)
        for index in register.all_idxs():
            array[index] = builder.join(
                returned[cursor : cursor + width], dtype=register.dtype
            )
            cursor += width
        outputs[register.name] = array if register.shape else array[()]
    prefix = returned[count:-1]
    target = returned[-1]
    assert isinstance(target, RegisterBinding)
    if uncompute:
        if junk:
            builder.free(builder.join(prefix))
        builder.free(target)
    else:
        if junk:
            outputs["junk"] = builder.join(prefix)
        outputs["target"] = target
    return _namespace_helpers(builder.finalize(outputs), name)
