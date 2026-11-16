"""Replay owned register workspace lifetimes on a Cirq qubit manager."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ...compiler.registers.register_binding import WorkspaceEvent
from ._qubit_manager import InteropQubitManager

if TYPE_CHECKING:
    from cirq import LineQubit, Operation, Qid, QubitManager


def _output_allocations(
    events: tuple[WorkspaceEvent, ...], outputs: Mapping[int, Qid]
) -> dict[int, int]:
    """Find the final clean lease for each transferred output wire."""
    pending = set(outputs)
    result: dict[int, int] = {}
    for index in range(len(events) - 1, -1, -1):
        if not pending:
            break
        event = events[index]
        wires = pending.intersection(event.wires)
        if not wires:
            continue
        if event.kind != "allocate" or event.dirty:
            raise ValueError("Cirq output transfer requires a final live clean lease")
        result.update((wire, index) for wire in wires)
        pending.difference_update(wires)
    if pending:
        raise ValueError("Cirq output transfer requires a final live clean lease")
    return result


def bind_workspace_operations(
    operations: Sequence[Operation],
    events: tuple[WorkspaceEvent, ...],
    inputs: Sequence[Qid],
    consumed_inputs: frozenset[int],
    output_groups: Sequence[Mapping[int, Qid]],
    base: QubitManager | None,
    *,
    instruction_offsets: Sequence[int] | None = None,
) -> tuple[Operation, ...]:
    """Remap emitted operations at physical lease boundaries."""
    outputs = {wire: qubit for group in output_groups for wire, qubit in group.items()}
    final_allocations = _output_allocations(events, outputs)
    manager = InteropQubitManager(base)
    manager.manage_qubits((*inputs, *outputs.values()))
    ports = dict(enumerate(inputs))
    pending_inputs = set(consumed_inputs)
    live: dict[int, Qid] = {}
    output_aliases: dict[int, Qid] = {}
    result: list[Operation] = []
    start = 0

    def locate(q: Qid) -> Qid:
        wire = cast("LineQubit", q).x
        if wire in ports:
            return ports[wire]
        if wire in output_aliases:
            return output_aliases[wire]
        return live[wire]

    try:
        for index, event in enumerate(events):
            offset = event.instruction_offset
            if offset is None or offset < 0:
                raise ValueError("Cirq workspace requires a valid instruction offset")
            if instruction_offsets is not None and offset >= len(instruction_offsets):
                raise ValueError("Cirq workspace event offset exceeds the native body")
            stop = (
                offset if instruction_offsets is None else instruction_offsets[offset]
            )
            if not start <= stop <= len(operations):
                raise ValueError(
                    "Cirq workspace events require ordered instruction offsets"
                )
            result.extend(op.transform_qubits(locate) for op in operations[start:stop])
            start = stop
            if event.kind == "allocate":
                if set(event.wires).intersection(ports.keys() | live.keys()):
                    raise ValueError(
                        "Cirq workspace may only lease private unbound wires"
                    )
                if not event.wires:
                    continue
                allocate = manager.qborrow if event.dirty else manager.qalloc
                supplied = allocate(len(event.wires))
                if len(supplied) != len(event.wires):
                    manager.qfree(supplied)
                    raise ValueError("Cirq manager returned the wrong workspace size")
                live.update(zip(event.wires, supplied, strict=True))
                for wire in event.wires:
                    if final_allocations.get(wire) == index:
                        output_aliases[wire] = outputs[wire]
            elif event.kind == "free":
                if event.wires:
                    if not set(event.wires) <= live.keys() | pending_inputs:
                        raise ValueError("Cirq workspace cannot free an unowned wire")
                    released = [
                        ports.pop(wire) if wire in pending_inputs else live.pop(wire)
                        for wire in event.wires
                    ]
                    pending_inputs.difference_update(event.wires)
                    manager.qfree(released)
            else:
                raise ValueError("Cirq workspace does not support retired wires")
        result.extend(op.transform_qubits(locate) for op in operations[start:])
        if pending_inputs:
            raise ValueError("Cirq workspace must release every owned left-only input")
        if live.keys() != outputs.keys():
            raise ValueError("Cirq workspace may only keep declared output leases live")
        for group in output_groups:
            if group:
                manager.qfree([live.pop(wire) for wire in group])
    except KeyError as exc:
        raise ValueError("Cirq workspace accessed a wire outside its lease") from exc
    finally:
        if live:
            manager.qfree(live.values())
    return tuple(result)
