# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 apply_gate_to_lth_target.py.

"""Store bounded selected Pauli calls through the shared unary traversal."""

from math import prod
from typing import Literal

from ....core.ir import Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program
from ..control.unary_wire_calls import (
    adjoint_unary_wire_program,
    append_nested_unary_wire_calls,
    bit_wire_call,
)


def selected_pauli_program(
    selection: tuple[Register, ...],
    pauli: Literal["X", "Y", "Z"] = "Z",
    *,
    control_width: int = 1,
    adjoint: bool = False,
) -> ProgramRecord:
    """Select one row-major Pauli using one or two positive control bits.

    Coordinates are concrete scalar bounded unsigned MSB-first words. Controls
    occupy one opaque word. Private conjunctions and predicates return clean.
    """
    if (
        type(control_width) is not int
        or control_width not in (1, 2)
        or pauli not in ("X", "Y", "Z")
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Stored selected Pauli requires one or two controls and a Pauli"
        )
    bounds: list[int] = []
    if not selection:
        raise ValueError("Stored selected Pauli requires selection words")
    for reg in selection:
        bound = dict(reg.dtype.parameters).get("bound")
        if (
            reg.shape_symbolic
            or reg.dtype.encoding != "uint"
            or type(reg.bitsize) is not int
            or reg.bitsize < 1
            or not isinstance(bound, int)
            or isinstance(bound, bool)
            or not 1 < bound <= 1 << reg.bitsize
        ):
            raise ValueError(
                "Stored selected Pauli requires concrete bounded scalar coordinates"
            )
        bounds.append(bound)
    signature = Signature(
        (
            Register("control", opaque_register_type(control_width)),
            *selection,
            Register("target", opaque_register_type(prod(bounds))),
        )
    )
    cursor = 0
    ports: dict[str, tuple[int, ...]] = {}
    for reg in signature:
        width = int(reg.bitsize)
        ports[reg.name] = tuple(range(cursor, cursor + width))
        cursor += width
    calls: list[WireCall] = []
    predicate = ports["control"][0]
    if control_width == 2:
        predicate = cursor
        cursor += 1
        calls.append(
            (
                and_component_program((1, 1), name="unary_and"),
                {"ctrl": ports["control"], "target": (predicate,)},
            )
        )
    axes = tuple(ports[reg.name] for reg in selection)
    work = tuple(range(cursor, cursor + sum(map(len, axes))))

    def leaf(index: int, enabled: int) -> None:
        calls.append(
            bit_wire_call(
                Instruction("c" + pauli.lower(), (enabled, ports["target"][index]))
            )
        )

    append_nested_unary_wire_calls(
        calls, axes, tuple(bounds), work, leaf, control=predicate
    )
    if control_width == 2:
        calls.append(
            (
                and_component_program((1, 1), name="unary_and", uncompute=True),
                {"ctrl": ports["control"], "target": (predicate,)},
            )
        )
    record = wire_component_program(signature, ports, ports, calls)
    return adjoint_unary_wire_program(record) if adjoint else record
