# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 SelectFirstQuantizationWithProj.

"""Keep projectile SELECT swap order, caller banks and explicit temporary lifetimes."""

from collections.abc import Mapping

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _elements


def projectile_selection_program(
    signature: Signature,
    children: Mapping[str, ProgramRecord],
    nuclear_width: SymbolicInt,
) -> ProgramRecord:
    """Compose the original SELECT calls without asserting opaque leaf cleanup."""
    if children.keys() != {
        "controlled_exchange",
        "axis_exchange",
        "electron_exchange",
        "kinetic",
        "potential",
    }:
        raise ValueError("Projectile SELECT requires its five source child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    n = signature.get_left("proj").bitsize
    p_width = signature.get_left("sys").bitsize
    p = np.array(
        [builder.allocate(dtype=opaque_register_type(n)) for _ in range(3)],
        dtype=object,
    )
    q = np.array(
        [builder.allocate(dtype=opaque_register_type(p_width)) for _ in range(3)],
        dtype=object,
    )
    rl: BindingCollection = builder.allocate(dtype=opaque_register_type(nuclear_width))
    flags = np.asarray(state["ham_ctrl"]).copy()
    projectile = np.asarray(state["proj"]).copy()

    def exchange_projectile() -> None:
        nonlocal p
        result = builder.call_dict(
            entries["controlled_exchange"],
            {
                "ctrl": np.array([flags[3]], dtype=object),
                "sel": state["i"],
                "targets": state["sys"],
                "junk": p,
            },
        )
        flags[3] = np.asarray(result["ctrl"])[0]
        state["i"], state["sys"] = result["sel"], result["targets"]
        p = np.asarray(result["junk"]).copy()
        for axis in range(3):
            result = builder.call_dict(
                entries["axis_exchange"],
                {"ctrl": flags[3], "x": projectile[axis], "y": p[axis]},
            )
            flags[3], projectile[axis], p[axis] = (
                result["ctrl"],
                result["x"],
                result["y"],
            )

    def exchange_electron() -> None:
        nonlocal q
        result = builder.call_dict(
            entries["electron_exchange"],
            {"sel": state["j"], "targets": state["sys"], "junk": q},
        )
        state["j"], state["sys"] = result["sel"], result["targets"]
        q = np.asarray(result["junk"]).copy()

    exchange_projectile()
    exchange_electron()
    kinetic = builder.call_dict(
        entries["kinetic"],
        {
            "flag_T": flags[0],
            "flag_mean": flags[1],
            "plus": state["plus_t"],
            "w": state["w"],
            "w_mean": state["w_mean"],
            "r": state["r"],
            "s": state["s"],
            "p": p,
        },
    )
    flags[0], flags[1], state["plus_t"] = (
        kinetic["flag_T"],
        kinetic["flag_mean"],
        kinetic["plus"],
    )
    state.update({name: kinetic[name] for name in ("w", "w_mean", "r", "s")})
    p = np.asarray(kinetic["p"]).copy()
    potential = builder.call_dict(
        entries["potential"],
        {
            "flag_tuv": flags[0],
            "flag_uv": flags[2],
            "l": state["l"],
            "rl": rl,
            "nu": np.array(
                [state[name] for name in ("nu_x", "nu_y", "nu_z")], dtype=object
            ),
            "p": p,
            "q": q,
        },
    )
    flags[0], flags[2] = potential["flag_tuv"], potential["flag_uv"]
    state["l"], rl = potential["l"], potential["rl"]
    p, q = np.asarray(potential["p"]).copy(), np.asarray(potential["q"]).copy()
    for index, name in enumerate(("nu_x", "nu_y", "nu_z")):
        state[name] = np.asarray(potential["nu"])[index]
    exchange_projectile()
    exchange_electron()
    for word in _elements((p, q)):
        builder.free(word)
    state["ham_ctrl"], state["proj"] = flags, projectile
    for word in _elements((rl,)):
        builder.free(word)
    return builder.finalize(state)
