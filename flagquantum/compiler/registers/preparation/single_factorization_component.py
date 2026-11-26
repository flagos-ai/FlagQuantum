# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from locked Qualtran v0.7.0 chemistry/sf/single_factorization.py.

"""Preserve the original single-factorization roots, including source wiring."""

from collections.abc import Mapping

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def single_factorization_one_body_program(
    signature: Signature, children: Mapping[str, ProgramRecord]
) -> ProgramRecord:
    """Compose nine source calls while retaining untouched system words."""
    if children.keys() != {"prepare", "unprepare", "hadamard", "swap", "select"}:
        raise ValueError("Single-factorization one-body requires five child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    prepare_ports = ("l", "p", "q", "succ_pq")
    state.update(
        builder.call_dict(entries["prepare"], {k: state[k] for k in prepare_ports})
    )

    def hadamard(port: str) -> None:
        state[port] = builder.call_dict(entries["hadamard"], {"q": state[port]})["q"]

    def swap() -> None:
        result = builder.call_dict(
            entries["swap"],
            {"ctrl": state["swap_pq"], "x": state["p"], "y": state["q"]},
        )
        state["swap_pq"], state["p"], state["q"] = (
            result["ctrl"],
            result["x"],
            result["y"],
        )

    hadamard("spin")
    hadamard("swap_pq")
    swap()
    select_ports = ("p", "q", "spin", "succ_pq", "succ_l")
    state.update(
        builder.call_dict(entries["select"], {k: state[k] for k in select_ports})
    )
    swap()
    hadamard("spin")
    hadamard("swap_pq")
    state.update(
        builder.call_dict(entries["unprepare"], {k: state[k] for k in prepare_ports})
    )
    return builder.finalize(state)


def single_factorization_block_program(
    signature: Signature, children: Mapping[str, ProgramRecord]
) -> ProgramRecord:
    """Retain the original five-call block and its ordered result permutation."""
    if children.keys() != {"prepare", "unprepare", "one_body", "reflection"}:
        raise ValueError("Single-factorization block requires four child owners")
    builder = ComponentProgramBuilder(signature)
    entries = {
        role: builder.import_program(child, entry=child.functions[0].name)
        for role, child in children.items()
    }
    state = dict(builder.inputs)
    flags = np.asarray(state.pop("ctrl")).copy()
    p, q = np.asarray(state.pop("pq"))
    rotations = np.asarray(state.pop("rot_aa")).copy()
    result = builder.call_dict(
        entries["prepare"],
        {
            "l": state["l"],
            "succ_l": flags[0],
            "l_ne_zero": flags[1],
            "rot_aa": rotations[0],
        },
    )
    state["l"], flags[0], flags[1], rotations[0] = (
        result["l"],
        result["succ_l"],
        result["l_ne_zero"],
        result["rot_aa"],
    )

    def one_body() -> None:
        nonlocal p, q
        result = builder.call_dict(
            entries["one_body"],
            {
                "succ_l": flags[0],
                "l_ne_zero": flags[1],
                "succ_pq": flags[2],
                "l": state["l"],
                "p": p,
                "q": q,
                "swap_pq": state["swap_pq"],
                "spin": state["spin"],
                "rot_aa": rotations[1],
                "sys": state["sys"],
            },
        )
        flags[0], flags[1], flags[2] = (
            result["succ_l"],
            result["l_ne_zero"],
            result["succ_pq"],
        )
        state["l"], p, q = result["l"], result["p"], result["q"]
        # Source bb.add returns signature order (rot_aa, swap_pq, spin).
        # Its assignment names are (swap_pq, spin, rot_aa[1]).
        state["swap_pq"], state["spin"], rotations[1] = (
            result["rot_aa"],
            result["swap_pq"],
            result["spin"],
        )
        state["sys"] = result["sys"]

    one_body()
    result = builder.call_dict(
        entries["reflection"],
        {
            "control": flags[0],
            "reg0_": p,
            "reg1_": q,
            "reg2_": state["swap_pq"],
            "reg3_": state["spin"],
        },
    )
    flags[0], p, q, state["swap_pq"], state["spin"] = (
        result["control"],
        result["reg0_"],
        result["reg1_"],
        result["reg2_"],
        result["reg3_"],
    )
    one_body()
    result = builder.call_dict(
        entries["unprepare"],
        {
            "l": state["l"],
            "succ_l": flags[0],
            "l_ne_zero": flags[1],
            "rot_aa": rotations[1],
        },
    )
    state["l"], flags[0], flags[1], rotations[1] = (
        result["l"],
        result["succ_l"],
        result["l_ne_zero"],
        result["rot_aa"],
    )
    return builder.finalize(
        {
            **state,
            "ctrl": flags,
            "pq": np.array([p, q], dtype=object),
            "rot_aa": rotations,
        }
    )
