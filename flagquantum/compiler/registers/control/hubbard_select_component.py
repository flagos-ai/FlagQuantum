# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Compose Hubbard SELECT with exact optional control and scalar ownership."""

from math import pi

from ....core.ir import Instruction
from ....core.ir.program import ProgramRecord
from ..component_adjoint import adjoint_component_calls
from ..component_wire_graph import WireCall, wire_component_program
from ..control.unary_wire_calls import bit_wire_call
from ..layout.hubbard_signatures import hubbard_signature
from ..lookup.hubbard_majorana_component import hubbard_majorana_program
from ..lookup.hubbard_spin_up_component import hubbard_spin_up_program
from ..primitives.controlled_swap_component import (
    controlled_swap_program,
)
from ..primitives.global_phase_component import scalar_phase_program


def hubbard_select_program(
    dimension: int, *, controlled: bool = False, adjoint: bool = False
) -> ProgramRecord:
    """Keep the source eleven calls, relative phase and private predicate work.

    The uncontrolled source scalar uses the existing clean private carrier.
    Positive control uses the original S-dagger and CZ instead.
    Selection is MSB-first; all caller-owned ports pass through both directions.
    """
    if type(adjoint) is not bool:
        raise ValueError("Hubbard SELECT direction must be boolean")
    if dimension == 1:
        raise ValueError("Original stored Hubbard SELECT refuses singleton coordinates")
    signature = hubbard_signature("select", dimension, controlled=controlled)
    cursor = 0
    ports: dict[str, tuple[int, ...]] = {}
    for reg in signature:
        size = int(reg.bitsize)
        ports[reg.name] = tuple(range(cursor, cursor + size))
        cursor += size
    width = (dimension - 1).bit_length()
    swap = controlled_swap_program(width)
    swap_bit = ProgramRecord(
        (next(f for f in controlled_swap_program(1).functions if f.name == "swap_bit"),)
    )
    majoranas = {
        axis: hubbard_majorana_program(dimension, axis, controlled=controlled)
        for axis in ("X", "Y")
    }
    spin = hubbard_spin_up_program(dimension, controlled=controlled)
    calls: list[WireCall] = []

    def majorana(axis: str, prefix: str, spin_name: str) -> None:
        calls.append(
            (
                majoranas[axis],
                {
                    **({"control": ports["control"]} if controlled else {}),
                    "x": ports[prefix + "_x"],
                    "y": ports[prefix + "_y"],
                    "spin": ports[spin_name],
                    "target": ports["target"],
                },
            )
        )

    def exchange(x: str, y: str, child: ProgramRecord) -> None:
        calls.append((child, {"ctrl": ports["V"], "x": ports[x], "y": ports[y]}))

    majorana("Y", "p", "alpha")
    exchange("p_x", "q_x", swap)
    exchange("p_y", "q_y", swap)
    exchange("alpha", "beta", swap_bit)
    majorana("X", "q", "beta")
    exchange("alpha", "beta", swap_bit)
    exchange("p_y", "q_y", swap)
    exchange("p_x", "q_x", swap)
    if controlled:
        phase = bit_wire_call(Instruction("sdg", ports["control"]))
        inverse_phase = bit_wire_call(Instruction("s", (0,)))[0]
        sign = bit_wire_call(Instruction("cz", (*ports["control"], *ports["U"])))
    else:
        phase = (scalar_phase_program(-pi / 2), {})
        inverse_phase = scalar_phase_program(pi / 2)
        sign = bit_wire_call(Instruction("z", ports["U"]))
    calls.extend((phase, sign))
    calls.append(
        (
            spin,
            {
                **({"control": ports["control"]} if controlled else {}),
                "V": ports["V"],
                "x": ports["q_x"],
                "y": ports["q_y"],
                "target": ports["target"],
            },
        )
    )
    root = wire_component_program(signature, ports, ports, calls)
    if not adjoint:
        return root
    pairs = [
        (swap, swap),
        (swap_bit, swap_bit),
        (
            spin,
            hubbard_spin_up_program(dimension, controlled=controlled, adjoint=True),
        ),
    ]
    for axis in ("X", "Y"):
        pairs.append(
            (
                majoranas[axis],
                hubbard_majorana_program(
                    dimension, axis, controlled=controlled, adjoint=True
                ),
            )
        )
    pairs.append((phase[0], inverse_phase))
    pairs.append((sign[0], sign[0]))
    return adjoint_component_calls(root, tuple(pairs))
