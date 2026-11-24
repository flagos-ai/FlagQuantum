# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Stage order follows Qualtran v0.7.0 reflections/reflection_using_prepare.py.

"""Compose exact PREPARE inverses, zero reflection and private junk lifetimes."""

from math import isclose

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.multi_control_component import multi_control_z_program
from ..primitives.global_phase import global_phase_exponent
from ..primitives.pauli_rotation_component import z_power_function


def prepared_reflection_program(
    prepare: ProgramRecord,
    unprepare: ProgramRecord,
    selection_names: tuple[str, ...],
    *,
    control_value: int | None = None,
    global_phase: complex = 1,
    scalar_phase: ProgramRecord | None = None,
) -> ProgramRecord:
    """Own junk words and call exact unprepare, mark-zero and prepare stages.

    Supplied first functions are entries with identical THRU ports. Selection
    ports form the public signature; all remaining ports are clean private
    junk, including row-major arrays. The caller owns exact inverse semantics.
    Uncontrolled nontrivial phases require an explicit zero-port scalar owner.
    Its carrier and cleanup are private to that owner, not the reflection target.
    """
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("Reflection control activation must be zero or one")
    if control_value is None and global_phase != 1 and scalar_phase is None:
        raise ValueError(
            "Registerless reflection phase requires an explicit scalar owner"
        )
    if scalar_phase is not None and (
        control_value is not None
        or global_phase == 1
        or scalar_phase.functions[0].registers != ()
    ):
        raise ValueError("Uncontrolled reflection scalar owner must have zero ports")
    if not isclose(abs(global_phase), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Reflection global phase must be a unit coefficient")
    ports = prepare.functions[0].registers
    if (
        ports is None
        or ports != unprepare.functions[0].registers
        or any(reg.side != RegisterSide.THRU for reg in ports)
    ):
        raise ValueError("Reflection requires matching through PREPARE inverse ports")
    by_name = {reg.name: reg for reg in ports}
    if len(set(selection_names)) != len(selection_names) or any(
        name not in by_name for name in selection_names
    ):
        raise ValueError("Reflection requires distinct declared selection names")
    selections = tuple(by_name[name] for name in selection_names)
    junk = tuple(reg for reg in ports if reg.name not in selection_names)
    bit = quantum_bit_type()
    x = bit_gate_function(
        "reflection_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    signature = Signature(
        (
            *((Register("control", bit),) if control_value is not None else ()),
            *selections,
        )
    )
    builder = ComponentProgramBuilder(signature, (x,) if not control_value else ())
    state: dict[str, BindingCollection] = {
        reg.name: builder.inputs[reg.name] for reg in selections
    }
    for reg in junk:
        storage = np.empty(reg.shape, dtype=object)
        for index in reg.all_idxs():
            storage[index] = builder.allocate(dtype=reg.dtype)
        state[reg.name] = storage if reg.shape else storage[()]
    state = builder.call_program(unprepare, state, entry=unprepare.functions[0].name)
    # The source Cirq-to-Bloq conversion creates the phase-target lifetime at
    # its first use, after PREPARE inverse, rather than at qalloc request time.
    target = builder.allocate() if control_value is None else builder.inputs["control"]
    if not control_value:
        target = builder.call_dict(x.name, {"q": target})["q"]
    if scalar_phase is not None:
        # The source Cirq conversion assigns the disconnected GlobalPhase its
        # instance after the first target flip, before MultiControlZ.
        builder.call_program(scalar_phase, {}, entry=scalar_phase.functions[0].name)
    controls: list[RegisterBinding] = []
    for reg in selections:
        for word in np.asarray(state[reg.name]).flat:
            assert isinstance(word, RegisterBinding)
            controls.extend([word] if reg.bitsize == 1 else builder.split(word))
    returned = builder.call_program(
        multi_control_z_program((0,) * len(controls)),
        {
            **({"controls": np.asarray(controls, dtype=object)} if controls else {}),
            "target": target,
        },
    )
    target = returned["target"]
    if global_phase != 1 and control_value is not None:
        gate = (
            bit_gate_function(
                "reflection_phase",
                (Register("q", bit),),
                CircuitIR(
                    1,
                    (Instruction({-1: "z", 1j: "s", -1j: "sdg"}[global_phase], (0,)),),
                ),
            )
            if global_phase in (-1, 1j, -1j)
            else z_power_function(
                "reflection_phase", global_phase_exponent(global_phase)
            )
        )
        builder.define(gate)
        target = builder.call_dict(gate.name, {"q": target})["q"]
    # Retain source EARLIEST ordering of a final open-control X after PREPARE
    # when a relative phase occupies the intervening target moment.
    delayed_flip = control_value == 0 and global_phase != 1
    if not control_value and not delayed_flip:
        target = builder.call_dict(x.name, {"q": target})["q"]
    bits = iter(np.asarray(returned.get("controls", ())).flat)
    for reg in selections:
        if type(reg.bitsize) is not int:
            raise ValueError("Reflection selection widths must be concrete")
        storage = np.empty(reg.shape, dtype=object)
        for index in reg.all_idxs():
            word = tuple(next(bits) for _ in range(reg.bitsize))
            storage[index] = (
                word[0] if reg.bitsize == 1 else builder.join(word, dtype=reg.dtype)
            )
        state[reg.name] = storage if reg.shape else storage[()]
    state = builder.call_program(prepare, state, entry=prepare.functions[0].name)
    if delayed_flip:
        target = builder.call_dict(x.name, {"q": target})["q"]
    for reg in junk:
        for word in np.asarray(state.pop(reg.name)).flat:
            assert isinstance(word, RegisterBinding)
            builder.free(word)
    if control_value is None:
        assert isinstance(target, RegisterBinding)
        builder.free(target)
    else:
        state["control"] = target
    return builder.finalize(state)
