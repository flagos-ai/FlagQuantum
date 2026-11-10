"""Bind finite register Programs, work lifetimes and outputs to physical IR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ...core.classical_control import validate_classical_control
from ...core.classical_dataflow import ClassicalXor
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import Operation, ProgramRecord, ValueRef, verify_program_record
from ...core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from ...core.ir.program.registers import RegisterSide
from .component_schedule import local_operation_order
from .layout.workspace import RegisterWorkspace, WorkspaceLease
from .program_calls import _expand


@dataclass(frozen=True)
class WorkspaceEvent:
    """One physical lifetime event, retaining the declared source dirty flag."""

    position: int
    kind: Literal["allocate", "free"]
    wires: tuple[int, ...]
    dirty: bool


@dataclass(frozen=True)
class BoundRegisterProgram:
    """Circuit IR, ordered quantum outputs, lifetimes and optional classical slots.

    Classical outputs use empty quantum tuples. When present, classical slot
    tuples align with every returned element and quantum elements have no slot.
    """

    circuit: CircuitIR
    outputs: tuple[tuple[int, ...], ...]
    events: tuple[WorkspaceEvent, ...]
    classical_outputs: tuple[tuple[int, ...], ...] = ()


def bind_register_program(
    record: ProgramRecord,
    inputs: tuple[tuple[int, ...], ...],
    *,
    clean: tuple[int, ...] = (),
    borrowed: tuple[int, ...] = (),
    entry: str = "main",
    allow_workspace_outputs: bool = False,
    input_ownership: tuple[tuple[int, bool], ...] = (),
) -> BoundRegisterProgram:
    """Bind quantum ports and work lifetimes to caller-supplied physical wires.

    Local connection order is scheduled before finite call expansion. Quantum
    entry ports and disjoint concrete input/clean/borrowed wires are required.
    Explicitly allowing workspace outputs transfers their live physical wires
    to the caller through ordered outputs. The caller owns subsequent cleanup
    and borrowed-state restoration. Input release requires explicit caller pool
    ownership through ``input_ownership``. Measurement-produced CBit outputs
    retain owned classical slots separately. Destructive measurement of pool-owned
    wires requires a separate retirement contract and is rejected.
    """
    if type(allow_workspace_outputs) is not bool:
        raise ValueError("workspace output policy requires a built-in boolean")
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid register Program: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown Program entry {entry!r}")
    function = functions[entry]
    if record.physical_context is not None or any(
        len(definition.blocks) != 1
        or definition.blocks[0].terminator != "return"
        or definition.blocks[0].edges
        or definition.blocks[0].return_values is None
        for definition in functions.values()
    ):
        raise ValueError("physical register binding requires a finite quantum Program")
    block = function.blocks[0]
    assert block.return_values is not None
    if any(
        value.kind not in {"qubit", "register"}
        for value in (*block.arguments, *block.return_values)
    ):
        raise ValueError("physical register binding requires quantum-only entry ports")
    widths = (
        tuple(
            executable_register_width(dtype)
            for dtype in register_port_types(function, RegisterSide.LEFT)
        )
        if function.registers is not None
        else (1,) * len(block.arguments)
    )
    if not isinstance(inputs, tuple) or any(
        not isinstance(word, tuple) for word in inputs
    ):
        raise ValueError("physical inputs require an immutable tuple of wire tuples")
    if tuple(map(len, inputs)) != widths:
        raise ValueError("physical input widths must match declared entry elements")
    if not isinstance(clean, tuple) or not isinstance(borrowed, tuple):
        raise ValueError("workspace pools require immutable wire tuples")
    all_wires = (*tuple(wire for word in inputs for wire in word), *clean, *borrowed)
    if any(type(wire) is not int or wire < 0 for wire in all_wires) or len(
        set(all_wires)
    ) != len(all_wires):
        raise ValueError(
            "input and workspace bindings require disjoint nonnegative integer wires"
        )
    input_wires = {wire for word in inputs for wire in word}
    if (
        not isinstance(input_ownership, tuple)
        or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or type(pair[0]) is not int
            or pair[0] not in input_wires
            or type(pair[1]) is not bool
            for pair in input_ownership
        )
        or len({pair[0] for pair in input_ownership}) != len(input_ownership)
    ):
        raise ValueError(
            "input ownership requires distinct (input wire, boolean dirty) pairs"
        )
    if not all_wires:
        raise ValueError("physical circuit binding requires at least one physical wire")
    scheduled = {}
    for name, definition in functions.items():
        body = definition.blocks[0]
        order = local_operation_order(definition, functions)
        scheduled[name] = replace(
            definition,
            blocks=(
                replace(
                    body, operations=tuple(body.operations[index] for index in order)
                ),
            ),
        )
    operations, returned = _expand(scheduled[entry].blocks[0], scheduled)
    binder = _PhysicalBinding(RegisterWorkspace(clean, borrowed=borrowed))
    for wire, dirty in input_ownership:
        binder.pool.adopt((wire,), dirty=dirty)
        binder.adopted.add(wire)
    binder.wires.update(zip(block.arguments, inputs, strict=True))
    for position, operation in enumerate(operations):
        binder.emit(position, operation)
    outputs = tuple(
        () if value in binder.bits else binder.wires[value] for value in returned
    )
    classical_outputs = (
        tuple(
            (binder.bits[value],) if value in binder.bits else () for value in returned
        )
        if any(value in binder.bits for value in returned)
        else ()
    )
    if binder.live and not allow_workspace_outputs:
        raise ValueError(
            "live workspace outputs require explicit caller ownership transfer"
        )
    # Zero-width logical ownership has already been checked by Core. Empty
    # physical leases have no wire to release after a split into an empty array.
    for lease in binder.empty_leases:
        binder.pool.free(lease)
    circuit = CircuitIR(max(all_wires, default=-1) + 1, tuple(binder.instructions))
    validate_classical_control(circuit.instructions, n_qubits=circuit.n_wires)
    return BoundRegisterProgram(
        circuit, outputs, tuple(binder.events), classical_outputs
    )


class _PhysicalBinding:
    def __init__(self, pool: RegisterWorkspace) -> None:
        self.pool = pool
        self.wires: dict[ValueRef, tuple[int, ...]] = {}
        self.scalars: dict[ValueRef, float] = {}
        self.bits: dict[ValueRef, int] = {}
        self.instructions: list[Instruction] = []
        self.events: list[WorkspaceEvent] = []
        self.live: set[int] = set()
        self.adopted: set[int] = set()
        self.empty_leases: list[WorkspaceLease] = []
        self.next_bit = 0

    def emit(self, position: int, operation: Operation) -> None:
        opcode = operation.opcode
        if opcode == "allocate_register":
            assert (
                operation.register_type is not None
                and operation.workspace_dirty is not None
            )
            lease = self.pool.allocate(
                executable_register_width(operation.register_type),
                dirty=operation.workspace_dirty,
            )
            self.wires[operation.results[0]] = lease.wires
            self.live.update(lease.wires)
            if not lease.wires:
                self.empty_leases.append(lease)
            self.events.append(
                WorkspaceEvent(
                    position, "allocate", lease.wires, operation.workspace_dirty
                )
            )
        elif opcode == "free_register":
            wires = self.wires[operation.operands[0]]
            if not set(wires) <= self.live | self.adopted:
                raise ValueError(
                    "freeing caller input wires requires explicit ownership transfer"
                )
            self.pool.free_wires(wires)
            self.live.difference_update(wires)
            self.adopted.difference_update(wires)
            assert operation.workspace_dirty is not None
            self.events.append(
                WorkspaceEvent(position, "free", wires, operation.workspace_dirty)
            )
        elif opcode == "split_register":
            wires = self.wires[operation.operands[0]]
            self.wires.update(
                (value, (wire,))
                for value, wire in zip(operation.results, wires, strict=True)
            )
        elif opcode == "join_register":
            self.wires[operation.results[0]] = tuple(
                wire for value in operation.operands for wire in self.wires[value]
            )
        elif opcode == "measure_register":
            wires = self.wires[operation.operands[0]]
            if set(wires) & (self.live | self.adopted):
                raise ValueError(
                    "destructive measurement of pool-owned workspace requires "
                    "a retirement ownership contract"
                )
            self.instructions.append(
                Instruction(
                    "measure",
                    wires,
                    metadata={"is_dynamic": True, "classical_bit": self.next_bit},
                )
            )
            self.bits[operation.results[0]] = self.next_bit
            self.next_bit += 1
        elif opcode == "constant":
            assert operation.literal is not None
            self.scalars[operation.results[0]] = operation.literal
        elif opcode == "discard_classical_register":
            del self.bits[operation.operands[0]]
        elif opcode == "read_classical_register":
            slot = self.bits[operation.operands[0]]
            self.bits.update((value, slot) for value in operation.results)
        elif opcode in {"bool_zero", "bool_one", "bool_xor", "parity_register"}:
            target = self.next_bit
            self.next_bit += 1
            sources = tuple(self.bits[value] for value in operation.operands)
            self.instructions.append(
                ClassicalXor(sources, target, opcode == "bool_one").instruction()
            )
            if opcode == "parity_register":
                self.bits.update(zip(operation.results[:-1], sources, strict=True))
            self.bits[operation.results[-1]] = target
        else:
            self._primitive(operation)

    def _primitive(self, operation: Operation) -> None:
        wires = tuple(
            wire for value in operation.operands for wire in self.wires[value]
        )
        metadata: dict[str, object] = {}
        if operation.opcode in {"measure", "reset"}:
            metadata["is_dynamic"] = True
        if operation.opcode == "measure":
            metadata["classical_bit"] = self.next_bit
            self.bits[operation.results[-1]] = self.next_bit
            self.next_bit += 1
        if operation.condition_clauses:
            clauses = tuple(
                sorted(
                    (
                        tuple(
                            sorted(
                                (self.bits[item.value], int(item.equals))
                                for item in clause
                            )
                        )
                        for clause in operation.condition_clauses
                    ),
                    key=lambda clause: (len(clause), clause),
                )
            )
            if len(clauses) == 1:
                metadata["conditions"] = clauses[0]
            else:
                metadata["condition_clauses"] = clauses
        self.instructions.append(
            Instruction(
                operation.opcode,
                wires,
                params={
                    name: self.scalars[value] for name, value in operation.parameters
                },
                metadata=metadata,
            )
        )
        quantum_results = tuple(
            value for value in operation.results if value.kind == "qubit"
        )
        self.wires.update(
            (value, (wire,)) for value, wire in zip(quantum_results, wires, strict=True)
        )
