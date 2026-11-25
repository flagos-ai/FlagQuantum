# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 chemistry/sparse/prepare.py.

"""Compose sparse PREPARE while lookup owns its directional output banks."""

from dataclasses import replace

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..bits.on_each import on_each_word_program
from ..comparison.ordering_component import less_equal_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_via_predicate
from ..component_gate_function import bit_gate_function
from ..control.control_predicate import ControlRegister
from ..primitives.controlled_swap_component import controlled_swap_program
from .sparse_chemistry_prepare_metadata import SPARSE_PREPARE_TARGETS
from .uniform_preparation_component import uniform_preparation_program

SPARSE_ALIAS_SWAPS = (
    ("alt_p", "p"),
    ("alt_q", "q"),
    ("alt_r", "r"),
    ("alt_s", "s"),
    ("flag_1b", "alt_flag_1b"),
)
SPARSE_SYMMETRY_SWAPS = (
    ("swap_pqrs", "p", "r"),
    ("swap_pqrs", "q", "s"),
    ("swap_pq", "p", "q"),
    ("swap_rs", "r", "s"),
)


def _zero_controlled_z(cz: ProgramRecord) -> ProgramRecord:
    bit = quantum_bit_type()
    z = bit_gate_function(
        "prep_z", (Register("q", bit),), CircuitIR(1, (Instruction("z", (0,)),))
    )

    def add(
        owner: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        returned = owner.call_dict(
            cz.functions[0].name, {"q1": control, "q2": values["q"]}
        )
        return returned["q1"], (returned["q2"],)

    builder = ComponentProgramBuilder(
        Signature(z.registers or ()), (z, *cz.functions), name="zero_z_entry"
    )
    entry = builder.finalize(builder.call_dict(z.name, builder.inputs))
    result = control_component_via_predicate(
        entry,
        {entry.functions[0].name: add},
        (Register("ctrl", bit),),
        (ControlRegister(1, (0,)),),
        entry=entry.functions[0].name,
    )
    return replace(
        result,
        functions=tuple(
            function for function in result.functions if function.name != z.name
        ),
    )


def sparse_prepare_component_pair(
    signature: Signature,
    row_count: int,
    lookup_pair: tuple[ProgramRecord, ProgramRecord],
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep source calls and typed sides; the inverse lookup owner is explicit.

    Row/alias/symmetry words and less_than are caller-owned THRU ports. The
    forward lookup supplies RIGHT target/bank words, consumed by its paired
    inverse. A native measured inverse is an explicit dependency, not a
    decomposition attributed to the source model-only erasure leaf.
    An explicit caller-bank lookup pair instead consumes and returns matching
    THRU bank words. That dependency must independently establish a unitary on
    arbitrary banks before this composition is used inside a reflection.
    """
    mu = signature.get_left("sigma").bitsize
    orbital = signature.get_right("p").bitsize
    if (
        type(row_count) is not int
        or row_count < 2
        or type(mu) is not int
        or mu < 1
        or type(orbital) is not int
        or orbital < 1
    ):
        raise ValueError("Sparse PREPARE requires concrete nonzero physical words")
    uniform = (
        uniform_preparation_program(row_count),
        uniform_preparation_program(row_count, adjoint=True),
    )
    bit = quantum_bit_type()
    h = bit_gate_function(
        "prep_h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    cz = ProgramRecord(
        (
            bit_gate_function(
                "prep_cz",
                (Register("q1", bit), Register("q2", bit)),
                CircuitIR(2, (Instruction("cz", (0, 1)),)),
            ),
        )
    )
    zero_z = _zero_controlled_z(cz)
    repeated = on_each_word_program(mu, h)
    inverse_repeated = adjoint_component_calls(
        repeated, ((ProgramRecord((h,)), ProgramRecord((h,))),)
    )
    compare = less_equal_program(mu)
    word_swap, flag_swap = controlled_swap_program(orbital), controlled_swap_program(1)
    pairs = (
        uniform,
        lookup_pair,
        (ProgramRecord((h,)), ProgramRecord((h,))),
        (cz, cz),
        (zero_z, zero_z),
        (repeated, inverse_repeated),
        (compare, compare),
        (word_swap, word_swap),
        (flag_swap, flag_swap),
    )
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    state["d"] = builder.call_program(uniform[0], {"target": state.pop("d")})["target"]
    lookup_inputs = {"selection": state.pop("d")}
    lookup_lefts = {
        reg.name
        for reg in Signature(lookup_pair[0].functions[0].registers or ()).lefts()
    }
    for index, name in enumerate(SPARSE_PREPARE_TARGETS):
        for port, outer in (
            (f"target{index}_", name),
            (f"junk_target{index}_", "jnk_" + name),
        ):
            if port in lookup_lefts:
                lookup_inputs[port] = state.pop(outer)
    loaded = builder.call_program(lookup_pair[0], lookup_inputs)
    state["d"] = loaded.pop("selection")
    for index, name in enumerate(SPARSE_PREPARE_TARGETS):
        state[name] = loaded.pop(f"target{index}_")
        if f"junk_target{index}_" in loaded:
            state["jnk_" + name] = loaded.pop(f"junk_target{index}_")
    if loaded:
        raise ValueError("Sparse PREPARE lookup has unexpected outputs")
    state["sigma"] = builder.call_program(repeated, {"q": state.pop("sigma")})["q"]
    result = builder.call_program(
        compare,
        {
            "x": state.pop("keep"),
            "y": state.pop("sigma"),
            "target": state.pop("less_than"),
        },
    )
    state.update(keep=result["x"], sigma=result["y"], less_than=result["target"])
    result = builder.call_program(
        cz,
        {"q1": state.pop("less_than"), "q2": state.pop("alt_theta")},
        entry=cz.functions[0].name,
    )
    state.update(less_than=result["q1"], alt_theta=result["q2"])
    result = builder.call_program(
        zero_z,
        {"ctrl": state.pop("less_than"), "q": state.pop("theta")},
        entry=zero_z.functions[0].name,
    )
    state.update(less_than=result["ctrl"], theta=result["q"])

    def swap(control: str, left: str, right: str) -> None:
        child = flag_swap if left == "flag_1b" else word_swap
        returned = builder.call_program(
            child,
            {"ctrl": state.pop(control), "x": state.pop(left), "y": state.pop(right)},
        )
        state.update(
            {control: returned["ctrl"], left: returned["x"], right: returned["y"]}
        )

    for left, right in SPARSE_ALIAS_SWAPS:
        swap("less_than", left, right)
    for name in ("swap_pq", "swap_rs", "swap_pqrs"):
        state[name] = builder.call_program(
            ProgramRecord((h,)), {"q": state.pop(name)}, entry=h.name
        )["q"]
    for control, left, right in SPARSE_SYMMETRY_SWAPS:
        swap(control, left, right)
    forward = builder.finalize(state)
    return forward, adjoint_component_calls(forward, pairs)
