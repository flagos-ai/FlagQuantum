# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 prepare_hubbard.py.

"""Store the square Hubbard preparation recipe and its exact source adjoint."""

from math import acos, sqrt

from ...compiler.registers.arithmetic.bounded_increment_component import (
    bounded_increment_program,
)
from ...compiler.registers.component_adjoint import adjoint_component_calls
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.component_wire_graph import WireCall, wire_component_program
from ...compiler.registers.control.conjunction_program import conjunction_program
from ...compiler.registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from ...compiler.registers.primitives.controlled_hadamard import controlled_hadamard_ir
from ...compiler.registers.primitives.controlled_swap_component import (
    controlled_swap_program,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from .hubbard_prepare import PrepareHubbard


def hubbard_prepare_program(
    prepare: PrepareHubbard, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain typed selection and through-junk ports, with private clean predicates.

    Coordinates are MSB first. temp remains caller-owned logical junk in both
    directions. The singleton original stored decomposition is unavailable.
    ModAddK keeps its original atomic boundary above native increment lowering.
    """
    if type(adjoint) is not bool:
        raise ValueError("Hubbard preparation direction must be boolean")
    if prepare.dimension == 1:
        raise ValueError("Original singleton Hubbard preparation has no stored body")
    width = (prepare.dimension - 1).bit_length()
    cursor = 0
    ports = {}
    for register in prepare.signature:
        size = int(register.total_qubits())
        ports[register.name] = tuple(range(cursor, cursor + size))
        cursor += size
    bit = quantum_bit_type()

    def gate(
        name: str, names: tuple[str, ...], instruction: Instruction
    ) -> ProgramRecord:
        return ProgramRecord(
            (
                bit_gate_function(
                    name,
                    tuple(Register(port, bit) for port in names),
                    CircuitIR(len(names), (instruction,)),
                ),
            )
        )

    h = gate("hubbard_h", ("q",), Instruction("h", (0,)))
    x = gate("hubbard_x", ("q",), Instruction("x", (0,)))
    cx = gate("hubbard_cx", ("ctrl", "target"), Instruction("cx", (0, 1)))
    ch = ProgramRecord(
        (
            bit_gate_function(
                "hubbard_ch",
                (Register("ctrl", bit), Register("target", bit)),
                CircuitIR(2, controlled_hadamard_ir((0,), 1, control_values=(1,))),
            ),
        )
    )
    angle = 2 * acos(sqrt(prepare.t * 2 * prepare.dimension**2 / prepare.alpha))
    conditional_angle = 2 * acos(sqrt(1 / 5))
    ry = gate("hubbard_ry", ("q",), Instruction("ry", (0,), params={"theta": angle}))
    inverse_ry = gate(
        "hubbard_ry_inverse", ("q",), Instruction("ry", (0,), params={"theta": -angle})
    )
    cry = gate(
        "hubbard_cry",
        ("ctrl", "q"),
        Instruction("cry", (0, 1), params={"theta": conditional_angle}),
    )
    inverse_cry = gate(
        "hubbard_cry_inverse",
        ("ctrl", "q"),
        Instruction("cry", (0, 1), params={"theta": -conditional_angle}),
    )
    uniform = uniform_preparation_program(prepare.dimension)
    swap = controlled_swap_program(width)
    direction_swap = controlled_swap_program(2 * width + 1)
    increment = bounded_increment_program(width, prepare.dimension, (0, 0))
    compute = conjunction_program((0, 0, 1), name="hubbard_direction")
    erase = conjunction_program(
        (0, 0, 1), name="hubbard_direction_erase", uncompute=True
    )
    u, v, alpha, beta = (ports[name] for name in ("U", "V", "alpha", "beta"))
    temp = ports["temp"]
    calls: list[WireCall] = [
        (ry, {"q": v}),
        (cry, {"ctrl": v, "q": u}),
        (uniform, {"target": ports["p_x"]}),
        (uniform, {"target": ports["p_y"]}),
        (h, {"q": temp[:1]}),
        (h, {"q": temp[1:]}),
        (cx, {"ctrl": u, "target": v}),
        (x, {"q": beta}),
        (x, {"q": v}),
        (ch, {"ctrl": v, "target": alpha}),
        (cx, {"ctrl": v, "target": beta}),
        (x, {"q": v}),
    ]
    left = (*ports["p_x"], *ports["p_y"], *alpha)
    right = (*ports["q_x"], *ports["q_y"], *beta)
    calls.extend(
        (cx, {"ctrl": (a,), "target": (b,)}) for a, b in zip(left, right, strict=True)
    )
    axis = {"ctrl": temp[:1], "x": ports["q_x"], "y": ports["q_y"]}
    predicate = {"ctrl": (*u, *v, temp[1]), "junk": (cursor + 1,), "target": (cursor,)}
    calls.extend(
        (
            (swap, axis),
            (increment, {"ctrl": (*u, *v), "x": ports["q_x"]}),
            (swap, axis),
            (compute, predicate),
            (direction_swap, {"ctrl": (cursor,), "x": left, "y": right}),
            (erase, predicate),
        )
    )
    record = wire_component_program(
        prepare.signature, ports, ports, calls, barriers=(12,)
    )
    if not adjoint:
        return record
    inverses = {
        id(ry): inverse_ry,
        id(cry): inverse_cry,
        id(uniform): uniform_preparation_program(prepare.dimension, adjoint=True),
        id(increment): bounded_increment_program(
            width, prepare.dimension, (0, 0), adjoint=True
        ),
        id(compute): erase,
        id(erase): compute,
    }
    return adjoint_component_calls(
        record, tuple((child, inverses.get(id(child), child)) for child, _ in calls)
    )
