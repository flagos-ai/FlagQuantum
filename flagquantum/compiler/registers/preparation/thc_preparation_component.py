# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls follow locked Qualtran v0.7.0 chemistry/thc/prepare.py.

"""Compose the complete source THC PREPARE with explicit directional children."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..bits.on_each import on_each_word_program
from ..comparison.ordering_component import less_equal_program
from ..component_adjoint import adjoint_component_calls, adjoint_component_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
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
    recipes: dict[str, tuple[ProgramRecord, ProgramRecord]] = {}

    def call(
        program: ProgramRecord,
        bindings: Mapping[str, object],
        inverse: ProgramRecord | None = None,
    ) -> dict[str, BindingCollection]:
        entry = builder.import_program(program, entry=program.functions[0].name)
        recipes.setdefault(entry, (program, program if inverse is None else inverse))
        return builder.call_dict(entry, bindings)

    names = ("mu", "nu", "succ", "nu_eq_mp1", "rot")
    prepared = call(
        uniform_pair[0],
        {name: state.pop(name) for name in names},
        inverse=uniform_pair[1],
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
        call(
            index_pair[0],
            {name: state.pop(name) for name in ("mu", "nu", "s")},
            inverse=index_pair[1],
        )
    )
    loaded = call(
        lookup_pair[0],
        {"selection": state.pop("s")},
        inverse=lookup_pair[1],
    )
    state["s"] = loaded.pop("selection")
    for i, name in enumerate(THC_PREPARE_TARGETS):
        state[name] = loaded.pop(f"target{i}_")
        if f"junk_target{i}_" not in loaded:
            raise ValueError("Original zero-block THC PREPARE omits junk_target0_")
        state["junk_" + name] = loaded.pop(f"junk_target{i}_")
    if loaded:
        raise ValueError("THC PREPARE lookup has unexpected output words")

    def one(
        program: ProgramRecord, name: str, inverse: ProgramRecord | None = None
    ) -> None:
        state[name] = call(program, {"q": state.pop(name)}, inverse)["q"]

    def less_equal() -> None:
        returned = call(
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
        returned = call(
            cz,
            {"q1": state.pop(left), "q2": state.pop(right)},
        )
        state[left], state[right] = returned["q1"], returned["q2"]

    def exchange(control: str, left: str, right: str) -> None:
        returned = call(
            swap,
            {"ctrl": state.pop(control), "x": state.pop(left), "y": state.pop(right)},
        )
        state[control], state[left], state[right] = (
            returned["ctrl"],
            returned["x"],
            returned["y"],
        )

    one(sigma, "sigma", inverse_sigma)
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
    returned = call(
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
    # Exact recipes were recorded at the importing call, avoiding repeated
    # normalization of the entire reachable QROM graph for semantic matching.
    definitions = list(forward.functions)
    mapping: dict[str, str] = {}
    for position, (callee, (child, inverse)) in enumerate(recipes.items()):
        if child == inverse:
            mapping[callee] = callee
            continue
        prefix = f"__fq_thc_inverse_{position}"
        while any(function.name.startswith(prefix) for function in definitions):
            prefix += "_"
        imported = _namespace_helpers(inverse, prefix)
        definitions.extend(imported.functions)
        mapping[callee] = imported.functions[0].name
    return forward, adjoint_component_program(
        replace(forward, functions=tuple(definitions)), mapping
    )
