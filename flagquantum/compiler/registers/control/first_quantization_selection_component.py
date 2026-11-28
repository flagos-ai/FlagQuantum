# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 SelectFirstQuantization.

"""Compose ordinary SELECT with its own exchange and temporary-register schedule."""

from collections.abc import Mapping

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _elements


def first_quantization_selection_program(
    signature: Signature,
    children: Mapping[str, ProgramRecord],
    nuclear_width: SymbolicInt,
) -> ProgramRecord:
    """Retain four source exchanges and through-owned selection/control words."""
    if children.keys() != {"exchange", "kinetic", "potential"}:
        raise ValueError("Ordinary SELECT requires its three source child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    width = signature.get_left("sys").bitsize
    p = np.array(
        [builder.allocate(dtype=opaque_register_type(width)) for _ in range(3)],
        dtype=object,
    )
    q = np.array(
        [builder.allocate(dtype=opaque_register_type(width)) for _ in range(3)],
        dtype=object,
    )
    rl: BindingCollection = builder.allocate(dtype=opaque_register_type(nuclear_width))

    def exchange(index: str, words: BindingCollection) -> BindingCollection:
        result = builder.call_dict(
            entries["exchange"],
            {"sel": state[index], "targets": state["sys"], "junk": words},
        )
        state[index], state["sys"] = result["sel"], result["targets"]
        return result["junk"]

    p = np.asarray(exchange("i", p))
    q = np.asarray(exchange("j", q))
    kinetic = builder.call_dict(
        entries["kinetic"],
        {
            "flag_T": state["tuv"],
            "plus": state["plus_t"],
            "w": state["w"],
            "r": state["r"],
            "s": state["s"],
            "p": p,
        },
    )
    state["tuv"], state["plus_t"] = kinetic["flag_T"], kinetic["plus"]
    state.update({name: kinetic[name] for name in ("w", "r", "s")})
    p = np.asarray(kinetic["p"])
    potential = builder.call_dict(
        entries["potential"],
        {
            "flag_tuv": state["tuv"],
            "flag_uv": state["uv"],
            "l": state["l"],
            "rl": rl,
            "nu": np.array(
                [state[name] for name in ("nu_x", "nu_y", "nu_z")], dtype=object
            ),
            "p": p,
            "q": q,
        },
    )
    state["tuv"], state["uv"], state["l"] = (
        potential["flag_tuv"],
        potential["flag_uv"],
        potential["l"],
    )
    rl = potential["rl"]
    for index, name in enumerate(("nu_x", "nu_y", "nu_z")):
        state[name] = np.asarray(potential["nu"])[index]
    p = np.asarray(exchange("i", potential["p"]))
    q = np.asarray(exchange("j", potential["q"]))
    for word in _elements((p, q, rl)):
        builder.free(word)
    return builder.finalize(state)
