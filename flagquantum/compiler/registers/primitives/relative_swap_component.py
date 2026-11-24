# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwapApprox.decompose_from_registers and its
# Cirq-style component import ordering. Physical segments retain their owner
# in relative_controlled_swap.py.

"""Retain relative-phase exchange child calls around the shared word fanout."""

from collections.abc import Sequence

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..bits.fanout_component import fanout_component_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .relative_controlled_swap import _relative_swap_parts, relative_swap_signature


def relative_swap_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Build the original finite immediate decomposition with no private work.

    All words remain caller-owned. The shared physical segments are emitted in
    source moment order, with a typed MultiTargetCNOT boundary between them.
    Width one keeps bit ports directly. Wider words split lazily in first-use
    order and rejoin across the fanout boundary, matching the source importer.
    """
    if type(adjoint) is not bool:
        raise ValueError("Relative swap direction must be boolean")
    if type(width) is not int or width < 1:
        raise ValueError("Relative swap requires a positive concrete integer width")
    bit = quantum_bit_type()
    leaves = tuple(
        bit_gate_function(
            gate,
            (
                tuple(Signature.build(ctrl=1, target=1))
                if gate == "cx"
                else (Register("q", bit),)
            ),
            CircuitIR(
                2 if gate == "cx" else 1,
                (Instruction(gate, (0, 1) if gate == "cx" else (0,)),),
            ),
        )
        for gate in ("cx", "s", "h", "tdg", "sdg", "t")
    )
    builder = ComponentProgramBuilder(relative_swap_signature(width), leaves)
    words = builder.inputs
    x, y = words["x"], words["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    ys = [y] if width == 1 else list(builder.split(y))
    xs = [x] if width == 1 else list(builder.split(x))
    before, after = _relative_swap_parts(
        tuple(range(width)), tuple(range(width, 2 * width)), layer_rotations=True
    )
    wires = [*xs, *ys]

    def emit(gates: Sequence[Instruction]) -> None:
        for gate in gates:
            ports = ("ctrl", "target") if gate.name == "cx" else ("q",)
            result = builder.call_dict(
                gate.name, dict(zip(ports, (wires[i] for i in gate.wires), strict=True))
            )
            for port, index in zip(ports, gate.wires, strict=True):
                value = result[port]
                assert isinstance(value, RegisterBinding)
                wires[index] = value

    emit(before)
    packed = wires[width] if width == 1 else builder.join(wires[width:])
    fanout = builder.call_program(
        fanout_component_program(width), {"control": words["ctrl"], "targets": packed}
    )
    y = fanout["targets"]
    assert isinstance(y, RegisterBinding)
    wires[width:] = [y] if width == 1 else list(builder.split(y))
    emit(after)
    record = builder.finalize(
        {
            "ctrl": fanout["control"],
            "x": wires[0] if width == 1 else builder.join(wires[:width]),
            "y": wires[width] if width == 1 else builder.join(wires[width:]),
        }
    )
    if not adjoint:
        return record
    gates = {function.name: ProgramRecord((function,)) for function in leaves}
    inverses = {"cx": "cx", "s": "sdg", "sdg": "s", "h": "h", "t": "tdg", "tdg": "t"}
    return adjoint_component_calls(
        record,
        (
            *((gates[gate], gates[inverse]) for gate, inverse in inverses.items()),
            (
                fanout_component_program(width),
                fanout_component_program(width, adjoint=True),
            ),
        ),
    )
