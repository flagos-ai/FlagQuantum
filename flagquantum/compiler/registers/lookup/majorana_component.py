# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 selected_majorana_fermion.py.

"""Store bounded Pauli Majorana traversal and its clean parity accumulator."""

from math import prod
from typing import Literal

from ....core.ir import Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_wire_graph import WireCall, wire_component_program
from ..control.unary_wire_calls import (
    adjoint_unary_wire_program,
    append_nested_unary_wire_calls,
    bit_wire_call,
)


def majorana_program(
    selection: tuple[Register, ...],
    pauli: Literal["X", "Y", "Z"] = "Y",
    *,
    control: Register | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Select a Pauli after a Jordan–Wigner prefix in C-order axis layout.

    Selection elements are scalar MSB-first bounded words. The optional control
    is one positive quantum bit. The private accumulator and all predicates
    start and finish clean. Exact source adjoints reverse stored calls.
    """
    if pauli not in ("X", "Y", "Z") or type(adjoint) is not bool:
        raise ValueError("Stored Majorana requires a Pauli and boolean direction")
    if not selection or any(
        reg.shape_symbolic or type(reg.bitsize) is not int or reg.bitsize < 1
        for reg in selection
    ):
        raise ValueError(
            "Stored Majorana requires nonempty concrete scalar selection words"
        )
    bounds: list[int] = []
    for reg in selection:
        bound = dict(reg.dtype.parameters).get("bound")
        if (
            reg.dtype.encoding != "uint"
            or not isinstance(bound, int)
            or isinstance(bound, bool)
            or not 1 < bound <= 1 << int(reg.bitsize)
        ):
            raise ValueError(
                "Stored Majorana requires bounded concrete iteration extents greater than one"
            )
        bounds.append(bound)
    extents = tuple(bounds)
    bit = quantum_bit_type()
    if control is not None and (control.dtype != bit or control.shape_symbolic):
        raise ValueError("Stored Majorana requires one scalar positive control bit")
    target_count = prod(extents)
    signature = Signature(
        (
            *((control,) if control is not None else ()),
            *selection,
            Register("target", opaque_register_type(target_count)),
        )
    )
    ports: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for register in signature:
        width = int(register.bitsize)
        ports[register.name] = tuple(range(cursor, cursor + width))
        cursor += width
    predicate = ports[control.name][0] if control is not None else None
    axes = tuple(ports[reg.name] for reg in selection)
    accumulator = cursor
    needed = sum(map(len, axes)) - int(control is None)
    work = tuple(range(cursor + 1, cursor + 1 + needed))
    calls: list[WireCall] = [
        bit_wire_call(
            Instruction("x", (accumulator,))
            if predicate is None
            else Instruction("cx", (predicate, accumulator))
        )
    ]

    def leaf(index: int, enabled: int) -> None:
        target = ports["target"][index]
        calls.extend(
            bit_wire_call(gate)
            for gate in (
                Instruction("cx", (enabled, accumulator)),
                Instruction("c" + pauli.lower(), (enabled, target)),
                Instruction("cz", (accumulator, target)),
            )
        )

    append_nested_unary_wire_calls(calls, axes, extents, work, leaf, control=predicate)
    record = wire_component_program(signature, ports, ports, calls)
    return adjoint_unary_wire_program(record) if adjoint else record
