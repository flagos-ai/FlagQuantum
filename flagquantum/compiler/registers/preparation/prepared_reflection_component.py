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
    signal_junk_names: tuple[str, ...] | None = None,
) -> ProgramRecord:
    """Compose through PREPARE pairs with explicit private or retained junk.

    Callers own exact inverse and private-cleanup proofs. Retained junk stays
    caller-owned; uncontrolled phases require an explicit scalar owner.
    """
    return _reflection_program(
        prepare,
        unprepare,
        selection_names,
        control_value=control_value,
        global_phase=global_phase,
        scalar_phase=scalar_phase,
        signal_junk_names=signal_junk_names,
        source_right_junk=False,
    )


def literal_prepared_reflection_program(
    prepare: ProgramRecord,
    unprepare: ProgramRecord,
    selection_names: tuple[str, ...],
    *,
    control_value: int | None = None,
    global_phase: complex = 1,
    scalar_phase: ProgramRecord | None = None,
) -> ProgramRecord:
    """Retain original RIGHT-junk lifetimes without asserting physical cleanup.

    The source allocates junk before the consuming PREPARE inverse, produces
    fresh RIGHT outputs in PREPARE and frees them afterwards. This is a logical
    source profile, not a unitary extension or an arbitrary-signal cleanup proof.
    Missing source leaf bodies must still refuse physical binding.
    """
    return _reflection_program(
        prepare,
        unprepare,
        selection_names,
        control_value=control_value,
        global_phase=global_phase,
        scalar_phase=scalar_phase,
        signal_junk_names=None,
        source_right_junk=True,
    )


def _reflection_program(
    prepare: ProgramRecord,
    unprepare: ProgramRecord,
    selection_names: tuple[str, ...],
    *,
    control_value: int | None = None,
    global_phase: complex = 1,
    scalar_phase: ProgramRecord | None = None,
    signal_junk_names: tuple[str, ...] | None = None,
    source_right_junk: bool,
) -> ProgramRecord:
    """Own junk words and call exact unprepare, mark-zero and prepare stages.

    Supplied first functions are entries with identical THRU ports. Selection
    ports form the public signature; all remaining ports are clean private
    junk, including row-major arrays. The caller owns exact inverse semantics.
    Uncontrolled nontrivial phases require an explicit zero-port scalar owner.
    Its carrier and cleanup are private to that owner, not the reflection target.
    Explicit signal_junk_names retain all junk as caller-owned THRU words.
    Private mode requires the caller to establish cleanup of the conjugated
    reflection; an exact PREPARE inverse alone does not prove that property.
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
    inverse_ports = unprepare.functions[0].registers
    if source_right_junk:
        if (
            ports is None
            or inverse_ports is None
            or any(reg.side == RegisterSide.LEFT for reg in ports)
            or tuple(Signature(inverse_ports).lefts())
            != tuple(Signature(ports).adjoint().lefts())
            or tuple(Signature(inverse_ports).rights())
            != tuple(Signature(ports).adjoint().rights())
        ):
            raise ValueError(
                "Literal reflection requires matching directional PREPARE ports"
            )
    elif (
        ports is None
        or ports != inverse_ports
        or any(reg.side != RegisterSide.THRU for reg in ports)
    ):
        raise ValueError("Reflection requires matching through PREPARE inverse ports")
    assert ports is not None
    by_name = {reg.name: reg for reg in ports}
    if len(set(selection_names)) != len(selection_names) or any(
        name not in by_name or by_name[name].side != RegisterSide.THRU
        for name in selection_names
    ):
        raise ValueError("Reflection requires distinct declared selection names")
    selections = tuple(by_name[name] for name in selection_names)
    junk = tuple(reg for reg in ports if reg.name not in selection_names)
    if signal_junk_names is not None and (
        len(signal_junk_names) != len(junk)
        or any(type(name) is not str or not name for name in signal_junk_names)
        or len(set(signal_junk_names)) != len(signal_junk_names)
        or set(signal_junk_names) & set(selection_names)
        or control_value is not None
        and "control" in signal_junk_names
    ):
        raise ValueError(
            "Signal junk requires distinct caller-owned names for every word"
        )
    retained = (
        tuple(
            Register(name, reg.dtype, shape=reg.shape, side=reg.side)
            for reg, name in zip(junk, signal_junk_names, strict=True)
        )
        if signal_junk_names is not None
        else ()
    )
    bit = quantum_bit_type()
    x = bit_gate_function(
        "reflection_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    signature = Signature(
        (
            *((Register("control", bit),) if control_value is not None else ()),
            *selections,
            *retained,
        )
    )
    builder = ComponentProgramBuilder(signature, (x,) if not control_value else ())
    state: dict[str, BindingCollection] = {
        reg.name: builder.inputs[reg.name] for reg in selections
    }
    for position, reg in enumerate(junk):
        if signal_junk_names is not None:
            state[reg.name] = builder.inputs[retained[position].name]
        else:
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
    # A registerless final PREPARE occupies the first Cirq moment. The source
    # converter therefore assigns it before the later mark-zero call whenever
    # an opening target X already occupies that moment.
    if not ports and control_value != 1:
        state = builder.call_program(prepare, state, entry=prepare.functions[0].name)
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
    if not ports and control_value == 1:
        # With no opening X, mark-zero is the first target operation. Empty
        # PREPARE follows it in that moment, before a relative phase gate.
        state = builder.call_program(prepare, state, entry=prepare.functions[0].name)
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
    if ports:
        state = builder.call_program(prepare, state, entry=prepare.functions[0].name)
    if delayed_flip:
        target = builder.call_dict(x.name, {"q": target})["q"]
    if signal_junk_names is not None:
        signal = {
            kept.name: state.pop(reg.name)
            for reg, kept in zip(junk, retained, strict=True)
        }
        state.update(signal)
    else:
        groups = (
            (
                tuple(reg for reg in junk if reg.side == RegisterSide.THRU),
                tuple(reg for reg in junk if reg.side == RegisterSide.RIGHT),
            )
            if source_right_junk
            else (junk,)
        )
        for position, group in enumerate(groups):
            for reg in group:
                for word in np.asarray(state.pop(reg.name)).flat:
                    assert isinstance(word, RegisterBinding)
                    builder.free(word)
            if source_right_junk and position == 0 and control_value is None:
                # Source Cirq capture consumes old RIGHT handles in the inverse.
                # Fresh RIGHT outputs follow the phase target in its live map.
                assert isinstance(target, RegisterBinding)
                builder.free(target)
    if control_value is None and not source_right_junk:
        assert isinstance(target, RegisterBinding)
        builder.free(target)
    elif control_value is not None:
        state["control"] = target
    return builder.finalize(state)
