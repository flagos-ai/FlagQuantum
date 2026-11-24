# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 phase_estimation/qubitization_qpe.py.

"""Compose the source reflection-sandwiched walk powers with typed ownership."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_wire_graph import WireCall, wire_component_program
from .phase_estimation_component import _through_ports


def qubitization_phase_estimation_program(
    preparation: ProgramRecord,
    controlled_walk: ProgramRecord,
    zero_reflection: ProgramRecord,
    walk_powers: tuple[ProgramRecord, ...],
    inverse_qft: ProgramRecord,
    target_registers: tuple[Register, ...],
    *,
    adjoint: bool = False,
    inverse_pairs: tuple[tuple[ProgramRecord, ProgramRecord], ...] = (),
) -> ProgramRecord:
    """Retain source root calls and exact child inverse recipes.

    Phase storage is a RIGHT MSB-first fractional word. Targets are unchanged
    caller-owned words, including any logical walk junk. Each supplied child
    owns its private work. Powers are explicit source-owned Programs for
    exponents 2, 4, 8, ...; no repetition or unitary semantics are inferred from
    their resource estimates. Only the lowest phase bit controls a walk; all
    other bits control zero reflections around an unconditional walk power.
    """
    if type(adjoint) is not bool:
        raise ValueError("Qubitization phase-estimation direction must be boolean")
    phase_ports = preparation.functions[0].registers or ()
    if (
        len(phase_ports) != 1
        or phase_ports[0].name != "qpe_reg"
        or phase_ports[0].side != RegisterSide.RIGHT
        or phase_ports[0].shape_symbolic
    ):
        raise ValueError("Walk phase preparation requires one scalar RIGHT qpe_reg")
    phase = phase_ports[0]
    width = phase.bitsize
    if type(width) is not int or width < 1 or len(walk_powers) != width - 1:
        raise ValueError("Walk phase estimation needs width-minus-one explicit powers")
    if (
        any(
            reg.side != RegisterSide.THRU or reg.is_symbolic() or reg.dtype.num_cbits
            for reg in target_registers
        )
        or len({reg.name for reg in target_registers}) != len(target_registers)
        or any(reg.name == phase.name for reg in target_registers)
    ):
        raise ValueError("Walk targets require distinct concrete quantum THRU ports")
    phase_wires = tuple(range(width))
    ports = {phase.name: phase_wires}
    cursor = width
    for reg in target_registers:
        size = int(reg.total_bits())
        ports[reg.name] = tuple(range(cursor, cursor + size))
        cursor += size
    targets = {reg.name: ports[reg.name] for reg in target_registers}

    def child_ports(
        child: ProgramRecord, control: tuple[str, int] | None = None
    ) -> dict[str, tuple[int, ...]]:
        registers = child.functions[0].registers
        if registers is None:
            raise ValueError("Walk dependency requires typed ports")
        available = dict(targets)
        if control is not None:
            name, bit = control
            if name in available:
                raise ValueError("Walk control name collides with a target port")
            available[name] = (bit,)
        if control is not None and control[0] not in {r.name for r in registers}:
            raise ValueError("Walk dependency lacks its explicit control port")
        if any(
            reg.name not in available
            or reg.side != RegisterSide.THRU
            or reg.is_symbolic()
            or reg.dtype.num_cbits
            or int(reg.total_bits()) != len(available[reg.name])
            for reg in registers
        ):
            raise ValueError("Walk dependency ports differ from their named bindings")
        return {reg.name: available[reg.name] for reg in registers}

    calls: list[WireCall] = [(preparation, {phase.name: phase_wires})]
    walk_ports = child_ports(controlled_walk, ("ctrl", phase_wires[-1]))
    if set(walk_ports) != {"ctrl", *targets}:
        raise ValueError("Controlled walk must retain all named target ports")
    calls.append((controlled_walk, walk_ports))
    for bit, power in zip(reversed(phase_wires[:-1]), walk_powers, strict=True):
        reflect_ports = child_ports(zero_reflection, ("control", bit))
        power_ports = child_ports(power)
        if set(power_ports) != set(targets):
            raise ValueError("Walk powers must retain all named target ports")
        calls.extend(
            (
                (zero_reflection, reflect_ports),
                (power, power_ports),
                (zero_reflection, reflect_ports),
            )
        )
    calls.append((inverse_qft, _through_ports(inverse_qft, phase_wires)))
    record = wire_component_program(
        Signature((*phase_ports, *target_registers)), targets, ports, calls
    )
    return adjoint_component_calls(record, inverse_pairs) if adjoint else record
