# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls follow locked Qualtran v0.7.0 chemistry/thc/prepare.py.

"""Compose the complete source THC PREPARE with explicit directional children."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..bits.on_each import on_each_word_program
from ..comparison.ordering_component import less_equal_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.multi_control_component import multi_control_x_program
from ..primitives.controlled_swap_component import controlled_swap_program
from .thc_preparation_metadata import THC_PREPARE_TARGETS


def thc_prepare_component_pair(
    signature: Signature,
    uniform_pair: tuple[ProgramRecord, ProgramRecord],
    index_pair: tuple[ProgramRecord, ProgramRecord],
    lookup_pair: tuple[ProgramRecord, ProgramRecord],
) -> tuple[ProgramRecord, ProgramRecord]:
    """Retain seventeen source calls, raw table order and the source flag choices.

    Caller-owned words are MSB-first; the lookup supplies RIGHT dataset/bank
    words, consumed by its explicit inverse. The literal uniform child retains
    its source defects and model-only GT. The source contiguous index and QROAM
    inverse have no executable source body. Providers must disclose any native
    extensions rather than attributing their gates to an upstream decomposition.
    """
    width, precision = (
        signature.get_left("mu").bitsize,
        signature.get_left("sigma").bitsize,
    )
    if (
        type(width) is not int
        or width < 1
        or type(precision) is not int
        or precision < 1
    ):
        raise ValueError("THC PREPARE requires concrete nonzero address/sampling words")
    bit = quantum_bit_type()
    h = ProgramRecord(
        (
            bit_gate_function(
                "thc_prepare_h",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("h", (0,)),)),
            ),
        )
    )
    x = ProgramRecord(
        (
            bit_gate_function(
                "thc_prepare_x",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("x", (0,)),)),
            ),
        )
    )
    cz = ProgramRecord(
        (
            bit_gate_function(
                "thc_prepare_cz",
                (Register("q1", bit), Register("q2", bit)),
                CircuitIR(2, (Instruction("cz", (0, 1)),)),
            ),
        )
    )
    sigma = on_each_word_program(precision, h.functions[0])
    inverse_sigma = adjoint_component_calls(sigma, ((h, h),))
    compare = less_equal_program(precision)
    swap = controlled_swap_program(width)
    predicate = multi_control_x_program((0, 1))
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    names = ("mu", "nu", "succ", "nu_eq_mp1", "rot")
    prepared = builder.call_program(
        uniform_pair[0],
        {name: state.pop(name) for name in names},
        entry=uniform_pair[0].functions[0].name,
    )
    # The source positional unpack swaps these two named uniform outputs.
    # Preserve that wiring in the literal profile, without inserting gates.
    state.update(
        mu=prepared["mu"],
        nu=prepared["nu"],
        rot=prepared["rot"],
        succ=prepared["nu_eq_mp1"],
        nu_eq_mp1=prepared["succ"],
    )
    state.update(
        builder.call_program(
            index_pair[0],
            {name: state.pop(name) for name in ("mu", "nu", "s")},
            entry=index_pair[0].functions[0].name,
        )
    )
    loaded = builder.call_program(
        lookup_pair[0],
        {"selection": state.pop("s")},
        entry=lookup_pair[0].functions[0].name,
    )
    state["s"] = loaded.pop("selection")
    for i, name in enumerate(THC_PREPARE_TARGETS):
        state[name] = loaded.pop(f"target{i}_")
        if f"junk_target{i}_" not in loaded:
            raise ValueError("Original zero-block THC PREPARE omits junk_target0_")
        state["junk_" + name] = loaded.pop(f"junk_target{i}_")
    if loaded:
        raise ValueError("THC PREPARE lookup has unexpected output words")

    def one(program: ProgramRecord, name: str) -> None:
        state[name] = builder.call_program(
            program, {"q": state.pop(name)}, entry=program.functions[0].name
        )["q"]

    def less_equal() -> None:
        returned = builder.call_program(
            compare,
            {
                "x": state.pop("keep"),
                "y": state.pop("sigma"),
                "target": state.pop("less_than"),
            },
        )
        state.update(
            keep=returned["x"], sigma=returned["y"], less_than=returned["target"]
        )

    def phase(left: str, right: str) -> None:
        returned = builder.call_program(
            cz,
            {"q1": state.pop(left), "q2": state.pop(right)},
            entry=cz.functions[0].name,
        )
        state[left], state[right] = returned["q1"], returned["q2"]

    def exchange(control: str, left: str, right: str) -> None:
        returned = builder.call_program(
            swap,
            {"ctrl": state.pop(control), "x": state.pop(left), "y": state.pop(right)},
        )
        state[control], state[left], state[right] = (
            returned["ctrl"],
            returned["x"],
            returned["y"],
        )

    one(sigma, "sigma")
    less_equal()
    phase("alt_theta", "less_than")
    one(x, "less_than")
    phase("less_than", "theta")
    one(x, "less_than")
    exchange("less_than", "alt_mu", "mu")
    exchange("less_than", "alt_nu", "nu")
    less_equal()
    for name in ("plus_a", "plus_b", "plus_mn"):
        one(h, name)
    returned = builder.call_program(
        predicate,
        {
            "controls": np.asarray(
                [state.pop("nu_eq_mp1"), state.pop("plus_a")], dtype=object
            ),
            "target": state.pop("extra_ctrl"),
        },
    )
    state["nu_eq_mp1"], state["plus_a"] = np.asarray(returned["controls"]).flat
    state["extra_ctrl"] = returned["target"]
    exchange("extra_ctrl", "mu", "nu")
    forward = builder.finalize(state)
    return forward, adjoint_component_calls(
        forward,
        (
            uniform_pair,
            index_pair,
            lookup_pair,
            (h, h),
            (x, x),
            (cz, cz),
            (sigma, inverse_sigma),
            (compare, compare),
            (swap, swap),
            (predicate, predicate),
        ),
    )
