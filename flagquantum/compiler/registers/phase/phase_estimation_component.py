# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow Qualtran v0.7.0 phase_estimation/text_book_qpe.py.

"""Compose a phase window, source controlled powers and inverse Fourier calls."""

from dataclasses import replace

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..bits.on_each import on_each_word_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_program
from ..component_gate_function import bit_gate_function
from ..component_power import power_word_program
from ..component_wire_graph import WireCall, wire_component_program
from ..primitives.controlled_rotation_component import controlled_z_power_program
from ..primitives.pauli_rotation_component import RotationAngle, z_power_function
from .textbook_qft_component import textbook_qft_program


def _through_ports(
    record: ProgramRecord, wires: tuple[int, ...]
) -> dict[str, tuple[int, ...]]:
    registers = record.functions[0].registers
    if registers is None or any(
        reg.side != RegisterSide.THRU or reg.is_symbolic() or reg.dtype.num_cbits
        for reg in registers
    ):
        raise ValueError("QPE child needs concrete quantum through ports")
    ports = {}
    cursor = 0
    for register in registers:
        size = int(register.total_bits())
        ports[register.name] = wires[cursor : cursor + size]
        cursor += size
    if cursor != len(wires):
        raise ValueError("QPE child width differs from its supplied wires")
    return ports


def phase_estimation_program(
    preparation: ProgramRecord,
    controlled_powers: tuple[ProgramRecord, ...],
    inverse_qft: ProgramRecord,
    target_registers: tuple[Register, ...],
    *,
    adjoint: bool = False,
    inverse_pairs: tuple[tuple[ProgramRecord, ProgramRecord], ...] = (),
) -> ProgramRecord:
    """Retain source power boundaries on reversed MSB-first phase bits.

    Preparation owns its RIGHT phase output. All target words remain caller
    owned. Each child owns private clean work and its exact inverse recipe.
    The inverse consumes a phase word in the prepared image; freeing it is a
    clean-storage promise, not a postselection on arbitrary phase inputs.
    """
    if type(adjoint) is not bool:
        raise ValueError("QPE direction must be boolean")
    phase_ports = preparation.functions[0].registers or ()
    if (
        len(phase_ports) != 1
        or phase_ports[0].name != "qpe_reg"
        or phase_ports[0].side != RegisterSide.RIGHT
    ):
        raise ValueError("QPE preparation requires one RIGHT qpe_reg")
    phase = phase_ports[0]
    width = phase.total_bits()
    if type(width) is not int or width < 1 or len(controlled_powers) != width:
        raise ValueError("QPE needs one controlled power per positive phase bit")
    if any(
        reg.side != RegisterSide.THRU or reg.is_symbolic() or reg.dtype.num_cbits
        for reg in target_registers
    ):
        raise ValueError("QPE targets require concrete quantum through registers")
    phase_wires = tuple(range(width))
    cursor = width
    ports = {phase.name: phase_wires}
    for register in target_registers:
        size = int(register.total_bits())
        ports[register.name] = tuple(range(cursor, cursor + size))
        cursor += size
    targets = tuple(
        wire for register in target_registers for wire in ports[register.name]
    )
    calls: list[WireCall] = [(preparation, {phase.name: phase_wires})]
    for bit, power in zip(reversed(phase_wires), controlled_powers, strict=True):
        calls.append((power, _through_ports(power, (bit, *targets))))
    calls.append((inverse_qft, _through_ports(inverse_qft, phase_wires)))
    inputs = {register.name: ports[register.name] for register in target_registers}
    record = wire_component_program(
        Signature((*phase_ports, *target_registers)), inputs, ports, calls
    )
    return adjoint_component_calls(record, inverse_pairs) if adjoint else record


def _z_powers(width: int, exponent: RotationAngle) -> tuple[ProgramRecord, ...]:
    controlled = controlled_z_power_program(exponent)

    def add_control(
        owner: ComponentProgramBuilder,
        ctrl: BindingCollection,
        inputs: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = np.asarray(
            owner.call_program(
                controlled, {"q": np.array([ctrl, inputs["q"]], dtype=object)}
            )["q"]
        )
        return result[0], (result[1],)

    phase = z_power_function("unitary", exponent)
    return (
        controlled,
        *(
            control_component_program(
                power_word_program(phase, 2**index), {"unitary": add_control}
            )
            for index in range(1, width)
        ),
    )


def rectangular_z_phase_estimation_program(
    preparation: ProgramRecord,
    exponent: RotationAngle,
    *,
    with_reverse: bool = True,
    qft_adjoint: bool = True,
    adjoint: bool = False,
) -> ProgramRecord:
    """Use the original rectangular/ZPow route, including unfused Power calls."""
    from ....core.ir.program.register_types import quantum_bit_type

    registers = preparation.functions[0].registers or ()
    if (
        len(registers) != 1
        or type(registers[0].bitsize) is not int
        or registers[0].bitsize < 1
    ):
        raise ValueError("Stored phase estimation requires a positive concrete width")
    width = registers[0].bitsize
    powers = _z_powers(width, exponent)
    inverse_qft = textbook_qft_program(
        width, with_reverse=with_reverse, adjoint=qft_adjoint
    )
    pairs: tuple[tuple[ProgramRecord, ProgramRecord], ...] = ()
    if adjoint:
        h = bit_gate_function(
            "hadamard",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("h", (0,)),)),
        )
        each = on_each_word_program(width, h)
        inverse_input = replace(
            preparation,
            functions=(
                replace(preparation.functions[0], name="main"),
                *preparation.functions[1:],
            ),
        )
        erase = adjoint_component_calls(inverse_input, ((each, each),))
        inverse_powers = _z_powers(width, -exponent)
        pairs = (
            (preparation, erase),
            *zip(powers, inverse_powers, strict=True),
            (
                inverse_qft,
                textbook_qft_program(
                    width, with_reverse=with_reverse, adjoint=not qft_adjoint
                ),
            ),
        )
    return phase_estimation_program(
        preparation,
        powers,
        inverse_qft,
        (Register("q", quantum_bit_type()),),
        adjoint=adjoint,
        inverse_pairs=pairs,
    )
