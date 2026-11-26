# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 chemistry/df/double_factorization.py.

"""Compose original DF roots without inventing their opaque physical leaves."""

from collections.abc import Mapping

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def double_factorization_one_body_program(
    signature: Signature, children: Mapping[str, ProgramRecord]
) -> ProgramRecord:
    """Retain nine source stages and caller-owned rotation metadata."""
    if children.keys() != {
        "prepare",
        "unprepare",
        "rotate",
        "unrotate",
        "hadamard",
        "swap",
        "phase",
    }:
        raise ValueError("Double-factorization one-body requires seven child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    system = np.asarray(state["sys"]).copy()
    prep_ports = ("xi", "offset", "rot", "succ_p", "p")
    rotation_ports = ("offset", "p", "rotations", "spin", "sys")
    state.update(
        builder.call_dict(entries["prepare"], {k: state[k] for k in prep_ports})
    )
    state["spin"] = builder.call_dict(entries["hadamard"], {"q": state["spin"]})["q"]

    def swap() -> None:
        nonlocal system
        result = builder.call_dict(
            entries["swap"], {"ctrl": state["spin"], "x": system[0], "y": system[1]}
        )
        state["spin"], system[0], system[1] = result["ctrl"], result["x"], result["y"]
        state["sys"] = system.copy()

    swap()
    state.update(
        builder.call_dict(entries["rotate"], {k: state[k] for k in rotation_ports})
    )
    system = np.asarray(state["sys"]).copy()
    phase = builder.call_dict(
        entries["phase"],
        {
            "ctrls": np.array([state["succ_l"], state["succ_p"]], dtype=object),
            "system": system[0],
        },
    )
    state["succ_l"], state["succ_p"] = np.asarray(phase["ctrls"])
    system[0] = phase["system"]
    state["sys"] = system.copy()
    state.update(
        builder.call_dict(entries["unrotate"], {k: state[k] for k in rotation_ports})
    )
    system = np.asarray(state["sys"]).copy()
    swap()
    state.update(
        builder.call_dict(entries["unprepare"], {k: state[k] for k in prep_ports})
    )
    state["spin"] = builder.call_dict(entries["hadamard"], {"q": state["spin"]})["q"]
    return builder.finalize(state)


def double_factorization_block_program(
    signature: Signature, children: Mapping[str, ProgramRecord]
) -> ProgramRecord:
    """Thread seven source stages and retain the original unused theta word."""
    if children.keys() != {
        "prepare",
        "unprepare",
        "load",
        "unload",
        "one_body",
        "reflection",
    }:
        raise ValueError("Double-factorization block requires six child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    flags = np.asarray(state.pop("ctrl")).copy()

    def outer(role: str) -> None:
        result = builder.call_dict(entries[role], {"l": state["l"], "succ_l": flags[0]})
        state["l"], flags[0] = result["l"], result["succ_l"]

    def indexed(role: str) -> None:
        result = builder.call_dict(
            entries[role],
            {
                "l": state["l"],
                "l_ne_zero": flags[1],
                "xi": state["xi"],
                "rot_data": state["rot"],
                "offset": state["offset"],
            },
        )
        state.update({k: result[k] for k in ("l", "xi", "offset")})
        flags[1], state["rot"] = result["l_ne_zero"], result["rot_data"]

    def one_body() -> None:
        ports = ("p", "rot_aa", "spin", "xi", "offset", "rot", "rotations", "sys")
        result = builder.call_dict(
            entries["one_body"],
            {
                "succ_l": flags[0],
                "l_ne_zero": flags[1],
                "succ_p": flags[3],
                **{k: state[k] for k in ports},
            },
        )
        state.update({k: result[k] for k in ports})
        flags[0], flags[1], flags[3] = (
            result["succ_l"],
            result["l_ne_zero"],
            result["succ_p"],
        )

    outer("prepare")
    indexed("load")
    one_body()
    result = builder.call_dict(
        entries["reflection"],
        {"control": flags[0], "reg0_": state["p"], "reg1_": state["spin"]},
    )
    flags[0], state["p"], state["spin"] = (
        result["control"],
        result["reg0_"],
        result["reg1_"],
    )
    one_body()
    indexed("unload")
    outer("unprepare")
    return builder.finalize({**state, "ctrl": flags})
