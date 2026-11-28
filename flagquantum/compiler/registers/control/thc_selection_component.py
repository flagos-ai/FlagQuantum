# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal composition follows Qualtran v0.7.0 chemistry/thc/select_bloq.py.

"""Retain THC SELECT's shared angle lifetime and original positive swap controls."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .thc_selection_control import control_thc_selection_program


def thc_selection_program(
    signature: Signature,
    rotations: tuple[FunctionRecord, FunctionRecord, FunctionRecord, FunctionRecord],
    system_swap: ProgramRecord,
    flag_swap: ProgramRecord,
    address_swap: ProgramRecord,
    phases: tuple[ProgramRecord, ProgramRecord],
    *,
    adjoint: bool = False,
    generic_control: bool = False,
) -> ProgramRecord:
    """Compose the complete source root with four explicitly directional leaves.

    The angle word is allocated once, passed through all rotation leaves and
    clean-freed. The source supplies no executable rotation body. An optional
    external control remains untouched, as in the original root. Caller words
    are THRU; predicate work belongs to the supplied phase/swap components.
    """
    builder = ComponentProgramBuilder(signature, rotations)
    state: dict[str, BindingCollection] = dict(builder.inputs)
    self_adjoint: set[str] = set()
    swaps: dict[str, ProgramRecord] = {}
    phase_roles: dict[str, tuple[ProgramRecord, tuple[int, ...]]] = {}

    def swap(program: ProgramRecord, control: str, left: str, right: str) -> None:
        entry = builder.import_program(program, entry=program.functions[0].name)
        self_adjoint.add(entry)
        swaps[entry] = program
        result = builder.call_dict(
            entry,
            {
                "ctrl": state.pop(control),
                "x": state.pop(left),
                "y": state.pop(right),
            },
        )
        state.update({control: result["ctrl"], left: result["x"], right: result["y"]})

    def rotate(function: FunctionRecord) -> None:
        result = builder.call_dict(
            function.name,
            {
                "nu_eq_mp1": state.pop("nu_eq_mp1"),
                "data": state.pop("data"),
                "sel": state.pop("mu"),
                "trg": state.pop("sys_a"),
            },
        )
        state.update(
            nu_eq_mp1=result["nu_eq_mp1"],
            data=result["data"],
            mu=result["sel"],
            sys_a=result["trg"],
        )

    def phase(program: ProgramRecord, controls: tuple[str, ...]) -> None:
        entry = builder.import_program(program, entry=program.functions[0].name)
        self_adjoint.add(entry)
        phase_roles[entry] = program, (1,) if len(controls) == 1 else (1, 0)
        result = builder.call_dict(
            entry,
            {
                "ctrls": tuple(state.pop(name) for name in controls),
                "system": state.pop("sys_b"),
            },
        )
        output = result["ctrls"]
        assert isinstance(output, np.ndarray)
        for i, name in enumerate(controls):
            state[name] = output[i]
        state["sys_b"] = result["system"]

    swap(system_swap, "plus_b", "sys_a", "sys_b")
    dtype = Signature(rotations[0].registers or ()).get_left("data").dtype
    state["data"] = builder.allocate(dtype=dtype)
    rotate(rotations[0])
    phase(phases[0], ("succ",))
    rotate(rotations[1])
    swap(system_swap, "plus_b", "sys_a", "sys_b")
    bit = quantum_bit_type()
    x = bit_gate_function(
        "thc_flip", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    state["plus_mn"] = builder.call_program(
        ProgramRecord((x,)), {"q": state.pop("plus_mn")}, entry=x.name
    )["q"]
    swap(flag_swap, "nu_eq_mp1", "plus_a", "plus_b")
    swap(address_swap, "nu_eq_mp1", "mu", "nu")
    swap(system_swap, "plus_b", "sys_a", "sys_b")
    rotate(rotations[2])
    phase(phases[1], ("succ", "nu_eq_mp1"))
    rotate(rotations[3])
    swap(system_swap, "plus_b", "sys_a", "sys_b")
    data = state.pop("data")
    assert isinstance(data, RegisterBinding)
    builder.free(data)
    record = builder.finalize(state)
    mapping = {name: name for name in (*self_adjoint, x.name)}
    mapping.update(
        {rotations[index].name: rotations[index ^ 1].name for index in range(4)}
    )
    operations = record.functions[0].blocks[0].operations
    # The source's shared angle allocation is its second call and its release
    # is last. Pair those actual imported identities, not equal opaque headers.
    allocate, release = operations[1].callee, operations[-1].callee
    assert allocate is not None and release is not None
    mapping[allocate], mapping[release] = release, allocate
    if adjoint:
        record = adjoint_component_program(record, mapping)
    if generic_control:
        return control_thc_selection_program(
            record, rotations, swaps, phase_roles, x.name, (allocate, release)
        )
    return record
