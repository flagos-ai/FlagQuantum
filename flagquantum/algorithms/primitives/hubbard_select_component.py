# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Compose the complete positive-controlled Hubbard SELECT stored graph."""

from ...compiler.registers.component_adjoint import adjoint_component_calls
from ...compiler.registers.component_wire_graph import WireCall, wire_component_program
from ...compiler.registers.control.unary_wire_calls import bit_wire_call
from ...compiler.registers.primitives.controlled_swap_component import (
    controlled_swap_program,
)
from ...core.ir import Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .hubbard_majorana import HubbardMajorana
from .hubbard_majorana_component import hubbard_majorana_program
from .hubbard_prepare import PrepareHubbard
from .hubbard_select import SelectHubbard
from .hubbard_spin_up import HubbardSpinUpZ
from .hubbard_spin_up_component import hubbard_spin_up_program


def hubbard_select_program(
    operation: SelectHubbard, *, adjoint: bool = False
) -> ProgramRecord:
    """Keep the source eleven calls, relative phase and private predicate work.

    Only the original successful positive-controlled stored path is admitted.
    The uncontrolled registerless phase retains its separate contract boundary.
    Selection is MSB-first; all caller-owned ports pass through both directions.
    """
    if type(adjoint) is not bool:
        raise ValueError("Hubbard SELECT direction must be boolean")
    if not operation.controlled or operation.control_value != 1:
        raise ValueError(
            "Stored Hubbard SELECT requires positive control; scalar and zero-control boundaries remain open"
        )
    if operation.dimension == 1:
        raise ValueError("Original stored Hubbard SELECT refuses singleton coordinates")
    signature = Signature(
        (
            Register("control", quantum_bit_type()),
            *PrepareHubbard(operation.dimension, 1.0, 4.0).selection_registers,
            Register("target", opaque_register_type(2 * operation.dimension**2)),
        )
    )
    cursor = 0
    ports: dict[str, tuple[int, ...]] = {}
    for reg in signature:
        size = int(reg.bitsize)
        ports[reg.name] = tuple(range(cursor, cursor + size))
        cursor += size
    width = (operation.dimension - 1).bit_length()
    swap = controlled_swap_program(width)
    swap_bit = ProgramRecord(
        (next(f for f in controlled_swap_program(1).functions if f.name == "swap_bit"),)
    )
    majoranas = {
        axis: hubbard_majorana_program(HubbardMajorana(operation.dimension, axis, True))
        for axis in ("X", "Y")
    }
    spin = hubbard_spin_up_program(HubbardSpinUpZ(operation.dimension, True))
    calls: list[WireCall] = []

    def majorana(axis: str, prefix: str, spin_name: str) -> None:
        calls.append(
            (
                majoranas[axis],
                {
                    "control": ports["control"],
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
    calls.append(bit_wire_call(Instruction("sdg", ports["control"])))
    calls.append(bit_wire_call(Instruction("cz", (*ports["control"], *ports["U"]))))
    calls.append(
        (
            spin,
            {
                "control": ports["control"],
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
            hubbard_spin_up_program(
                HubbardSpinUpZ(operation.dimension, True), adjoint=True
            ),
        ),
    ]
    for axis in ("X", "Y"):
        pairs.append(
            (
                majoranas[axis],
                hubbard_majorana_program(
                    HubbardMajorana(operation.dimension, axis, True), adjoint=True
                ),
            )
        )
    pairs.append((calls[-3][0], bit_wire_call(Instruction("s", (0,)))[0]))
    pairs.append((calls[-2][0], calls[-2][0]))
    return adjoint_component_calls(root, tuple(pairs))
