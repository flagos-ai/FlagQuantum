# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source-ordered calls adapted from locked Qualtran v0.7.0 unary_iteration_bloq.py.

"""Emit typed unary calls before the shared earliest-layer wire reconstruction."""

from collections.abc import Callable

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_adjoint import adjoint_component_calls
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import WireCall
from ..lookup.unary_iteration import _walk_nested_bounded_unary_tree
from .conjunction_program import and_component_program


def bit_wire_call(gate: Instruction) -> WireCall:
    """Bind an existing one- or two-bit primitive with source scalar port names."""
    if len(gate.wires) not in (1, 2):
        raise ValueError("Unary primitive calls require one or two wires")
    names = (
        ("q",)
        if len(gate.wires) == 1
        else ("q1", "q2") if gate.name == "cz" else ("ctrl", "target")
    )
    registers = (
        (Register("q", quantum_bit_type(), (2,)),)
        if gate.name == "cy"
        else tuple(Register(name, quantum_bit_type()) for name in names)
    )
    function = bit_gate_function(
        "unary_" + gate.name,
        registers,
        CircuitIR(
            len(names), (Instruction(gate.name, tuple(range(len(names))), gate.params),)
        ),
    )
    if gate.name == "cy":
        return ProgramRecord((function,)), {"q": gate.wires}
    return ProgramRecord((function,)), {
        name: (wire,) for name, wire in zip(names, gate.wires, strict=True)
    }


def append_nested_unary_wire_calls(
    calls: list[WireCall],
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    workspace: tuple[int, ...],
    emit_leaf: Callable[[int, int], None],
    *,
    control: int | None = None,
) -> None:
    """Retain source And compute/erase boundaries and caller-supplied leaf calls.

    Coordinates and workspace are distinct logical wire identities. And targets
    are directional clean outputs and inputs. Reuse of a consumed target owns a
    new lifetime. The existing wire composer schedules and binds those calls.
    """

    def emit_and(
        left: int, right: int, target: int, values: tuple[int, int], erase: bool
    ) -> None:
        program = and_component_program(values, name="unary_and", uncompute=erase)
        calls.append((program, {"ctrl": (left, right), "target": (target,)}))

    _walk_nested_bounded_unary_tree(
        control,
        axes,
        extents,
        workspace,
        emit_leaf,
        lambda gate: calls.append(bit_wire_call(gate)),
        emit_and,
    )


def adjoint_unary_wire_program(
    record: ProgramRecord,
    components: tuple[tuple[ProgramRecord, ProgramRecord], ...] = (),
) -> ProgramRecord:
    """Reverse stored Pauli calls with exact directional And erasure bodies."""
    pairs = list(components)
    for gate, wires in (("x", (0,)), ("cx", (0, 1)), ("cy", (0, 1)), ("cz", (0, 1))):
        child = bit_wire_call(Instruction(gate, wires))[0]
        pairs.append((child, child))
    for values in ((1, 0), (1, 1)):
        compute = and_component_program(values, name="unary_and")
        erase = and_component_program(values, name="unary_and", uncompute=True)
        pairs.extend(((compute, erase), (erase, compute)))
    return adjoint_component_calls(record, tuple(pairs))
