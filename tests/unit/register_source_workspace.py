"""Bind original source lifetime callbacks with an independent finite pool."""

from __future__ import annotations

from collections.abc import Iterable
from math import pi
from typing import Any, Literal

import numpy as np

from flagquantum.core.ir import Instruction
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

LifetimeEvent = tuple[Literal["allocate", "free"], tuple[int, ...], bool]


class SourceWorkspace:
    """Supply Cirq qubits to actual source qalloc/qborrow/qfree callbacks."""

    def __init__(self, clean: tuple[int, ...], borrowed: tuple[int, ...]) -> None:
        self.available = {False: list(reversed(clean)), True: list(reversed(borrowed))}
        self.live: dict[int, bool] = {}
        self.events: list[LifetimeEvent] = []

    def _allocate(self, n: int, dim: int, dirty: bool) -> list[Any]:
        import cirq

        assert dim == 2 and n <= len(self.available[dirty])
        wires = tuple(self.available[dirty].pop() for _ in range(n))
        assert not self.live.keys() & set(wires)
        self.live.update(dict.fromkeys(wires, dirty))
        self.events.append(("allocate", wires, dirty))
        return [cirq.LineQubit(wire) for wire in wires]

    def qalloc(self, n: int, dim: int = 2) -> list[Any]:
        return self._allocate(n, dim, False)

    def qborrow(self, n: int, dim: int = 2) -> list[Any]:
        return self._allocate(n, dim, True)

    def qfree(self, qubits: Iterable[Any]) -> None:
        wires = tuple(int(q.x) for q in qubits)
        assert wires and len(set(wires)) == len(wires)
        flags = tuple(self.live.pop(wire) for wire in wires)
        assert len(set(flags)) == 1
        self.available[flags[0]].extend(reversed(wires))
        self.events.append(("free", wires, flags[0]))


def source_workspace_gates(
    source: Any,
    ports: SourcePorts,
    clean: tuple[int, ...],
    borrowed: tuple[int, ...],
) -> tuple[tuple[Instruction, ...], tuple[LifetimeEvent, ...]]:
    """Read original graphs and allocate classical slots per measurement event.

    No native instruction, lifetime event or measurement ID enters this oracle.
    Repeated measurements of one physical wire receive separate event slots.
    """
    import cirq
    from qualtran.bloqs.basic_gates import (
        CNOT,
        CHadamard,
        Hadamard,
        Toffoli,
        XGate,
        ZPowGate,
    )
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.mcmt import And

    pool = SourceWorkspace(clean, borrowed)
    slot = 0

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal slot
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        if isinstance(bloq, (Allocate, Free, And)):
            quregs = {
                reg.name: np.array(
                    [cirq.LineQubit(w) for w in inputs[reg.name]], dtype=object
                ).reshape((*reg.shape, int(reg.bitsize)))
                for reg in bloq.signature.lefts()
            }
            _, returned = bloq.as_cirq_op(pool, **quregs)
            outputs = {
                name: tuple(int(q.x) for q in value.flat)
                for name, value in returned.items()
            }
            if not isinstance(bloq, And):
                return (), outputs
            target = inputs["target"] if bloq.uncompute else outputs["target"]
            gates = reference_arithmetic_gates(
                bloq,
                {**inputs, "target": target},
                (),
                {target[0]: slot},
                cirq_style=True,
            )
            if bloq.uncompute:
                slot += 1
            return gates, outputs
        if isinstance(bloq, Toffoli):
            return (
                reference_arithmetic_gates(
                    bloq.as_composite_bloq(), inputs, (), {}, keep_exchange_leaves=True
                ),
                inputs,
            )
        if isinstance(bloq, CHadamard):
            return (Instruction("ch", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, Hadamard):
            return (Instruction("h", inputs["q"]),), inputs
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, ZPowGate):
            return (
                Instruction(
                    "phase", inputs["q"], params={"theta": pi * float(bloq.exponent)}
                ),
            ), inputs
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and not pool.live
    assert [g.metadata["classical_bit"] for g in gates if g.name == "measure"] == list(
        range(slot)
    )
    return gates, tuple(pool.events)
