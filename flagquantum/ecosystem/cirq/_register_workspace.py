"""Replay owned register workspace lifetimes on a Cirq qubit manager."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ...compiler.registers.register_binding import WorkspaceEvent
from ._qubit_manager import InteropQubitManager

if TYPE_CHECKING:
    from cirq import LineQubit, Operation, Qid, QubitManager


def bind_workspace_operations(
    operations: Sequence[Operation],
    events: tuple[WorkspaceEvent, ...],
    qubits: Sequence[Qid],
    base: QubitManager | None,
) -> tuple[Operation, ...]:
    """Remap a one-operation-per-instruction stream at physical lease boundaries."""
    manager = InteropQubitManager(base)
    manager.manage_qubits(qubits)
    ports = dict(enumerate(qubits))
    live: dict[int, Qid] = {}
    result: list[Operation] = []
    start = 0

    def locate(q: Qid) -> Qid:
        wire = cast("LineQubit", q).x
        if wire in ports:
            return ports[wire]
        return live[wire]

    try:
        for event in events:
            stop = event.instruction_offset
            if stop is None or not start <= stop <= len(operations):
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
            elif event.kind == "free":
                if event.wires:
                    if not set(event.wires) <= live.keys():
                        raise ValueError("Cirq workspace cannot free an inactive lease")
                    manager.qfree([live.pop(wire) for wire in event.wires])
            else:
                raise ValueError("Cirq workspace does not support retired wires")
        result.extend(op.transform_qubits(locate) for op in operations[start:])
        if live:
            raise ValueError("Cirq workspace must release every private lease")
    except KeyError as exc:
        raise ValueError("Cirq workspace accessed a wire outside its lease") from exc
    finally:
        if live:
            manager.qfree(live.values())
    return tuple(result)
