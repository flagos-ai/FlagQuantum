# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 selected_majorana_fermion.py.

"""Store bounded Majorana traversal and its clean parity accumulator."""

from math import prod
from typing import Literal

from ....core.ir import Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program, conjunction_program
from ..control.unary_wire_calls import (
    adjoint_unary_wire_program,
    append_nested_unary_wire_calls,
    bit_wire_call,
)


def _through_wire_call(record: ProgramRecord, wires: tuple[int, ...]) -> WireCall:
    """Bind a finite owned gate by its source-ordered through ports."""
    registers = record.functions[0].registers
    if registers is None or any(
        reg.side != RegisterSide.THRU or reg.dtype.num_cbits or reg.is_symbolic()
        for reg in registers
    ):
        raise ValueError(
            "Majorana gate children require concrete quantum through ports"
        )
    cursor = 0
    ports = {}
    for register in registers:
        size = int(register.total_bits())
        ports[register.name] = wires[cursor : cursor + size]
        cursor += size
    if cursor != len(wires):
        raise ValueError("Majorana gate child width differs from its bound wires")
    return record, ports


def majorana_program(
    selection: tuple[Register, ...],
    pauli: Literal["X", "Y", "Z"] = "Y",
    *,
    control: Register | None = None,
    adjoint: bool = False,
    target_pair: tuple[ProgramRecord, ProgramRecord] | None = None,
    initial_pair: tuple[ProgramRecord, ProgramRecord] | None = None,
) -> ProgramRecord:
    """Select an owned target gate after a Jordan–Wigner prefix in C-order.

    Selection elements are scalar MSB-first bounded words. The optional control
    is one positive quantum word. Explicit gate pairs extend the existing
    Pauli recipe without importing an SDK. The private accumulator and all
    predicates start and finish clean. Exact source adjoints reverse calls.
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
            or not 1 <= bound <= 1 << int(reg.bitsize)
        ):
            raise ValueError(
                "Stored Majorana requires bounded concrete iteration extents"
            )
        bounds.append(bound)
    extents = tuple(bounds)
    if control is not None and (
        control.shape_symbolic
        or control.side != RegisterSide.THRU
        or control.dtype.num_cbits
        or type(control.bitsize) is not int
        or control.bitsize < 1
    ):
        raise ValueError("Stored Majorana requires one concrete quantum control word")
    count = int(control.bitsize) if control is not None else 0
    if not count and bounds[0] == 1:
        raise ValueError("Source uncontrolled Majorana refuses a singleton first axis")
    if count > 1 and initial_pair is None:
        raise ValueError("Multi-bit Majorana needs an exact accumulator initializer")
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
    controls = ports[control.name] if control is not None else ()
    predicate = controls[0] if controls else None
    axes = tuple(ports[reg.name] for reg in selection)
    accumulator = cursor
    start = cursor + 1 + max(0, count - 1)
    needed = sum(map(len, axes)) - int(not count)
    work = tuple(range(start, start + needed))
    calls: list[WireCall] = [
        (
            _through_wire_call(initial_pair[0], (*controls, accumulator))
            if initial_pair is not None
            else bit_wire_call(
                Instruction("x", (accumulator,))
                if predicate is None
                else Instruction("cx", (predicate, accumulator))
            )
        )
    ]
    pairs = [pair for pair in (target_pair, initial_pair) if pair is not None]
    reduction: WireCall | None = None
    if count > 1:
        compute = (
            and_component_program((1, 1), name="control_and")
            if count == 2
            else conjunction_program((1,) * count, name="control_and")
        )
        erase = (
            and_component_program((1, 1), name="control_and", uncompute=True)
            if count == 2
            else conjunction_program((1,) * count, name="control_and", uncompute=True)
        )
        predicate = start - 1
        reduction_ports = {"ctrl": controls, "target": (predicate,)}
        if count > 2:
            reduction_ports["junk"] = tuple(range(cursor + 1, predicate))
        calls.append((compute, reduction_ports))
        reduction = erase, reduction_ports
        pairs.extend(((compute, erase), (erase, compute)))

    def leaf(index: int, enabled: int) -> None:
        target = ports["target"][index]
        calls.append(bit_wire_call(Instruction("cx", (enabled, accumulator))))
        calls.append(
            _through_wire_call(target_pair[0], (enabled, target))
            if target_pair is not None
            else bit_wire_call(Instruction("c" + pauli.lower(), (enabled, target)))
        )
        calls.append(bit_wire_call(Instruction("cz", (accumulator, target))))

    append_nested_unary_wire_calls(calls, axes, extents, work, leaf, control=predicate)
    if reduction is not None:
        calls.append(reduction)
    record = wire_component_program(signature, ports, ports, calls)
    return adjoint_unary_wire_program(record, tuple(pairs)) if adjoint else record
