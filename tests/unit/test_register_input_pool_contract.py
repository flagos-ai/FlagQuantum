"""Original-source witnesses for explicitly pre-owned physical input wires."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("free_dirty", [False, True])
def test_source_preowned_input_returns_to_manager_before_output_allocation(
    width: int, dirty: bool, free_dirty: bool
) -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, Register, Side, Signature
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    events = []

    class Manager(cirq.GreedyQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            qubits = super().qalloc(n, dim)
            events.append(("allocate", tuple(qubits)))
            return qubits

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            qubits = super().qborrow(n, dim)
            events[-1] = ("borrow", tuple(qubits))
            return qubits

        def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
            wires = tuple(qubits)
            super().qfree(wires)
            events.append(("free", wires))

    manager = Manager("preowned", size=width, maximize_reuse=False)
    inputs = tuple(manager.qborrow(width) if dirty else manager.qalloc(width))
    events.clear()
    builder, arguments = BloqBuilder.from_signature(
        Signature(
            [
                Register("input", QAny(width), side=Side.LEFT),
                Register("out", QAny(width), side=Side.RIGHT),
            ]
        )
    )
    reversed_input = builder.join(builder.split(arguments["input"])[::-1])
    builder.free(reversed_input, dirty=free_dirty)
    out = builder.allocate(dtype=QAny(width), dirty=dirty)
    composite = builder.finalize(out=out)
    circuit, outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=manager, input=np.array(inputs)
    )
    assert tuple(circuit.all_operations()) == ()
    assert events == [
        ("free", inputs[::-1]),
        ("borrow" if dirty else "allocate", inputs[::-1]),
    ]
    assert tuple(outputs["out"]) == inputs[::-1]
    assert get_cost_value(composite, QECGatesCost()).asdict() == {}
    assert get_cost_value(composite, QubitCount()) == width
