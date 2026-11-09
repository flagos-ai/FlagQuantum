"""Original-source ownership witnesses for right-only physical register ports."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("dirty", [False, True])
def test_source_allocated_output_retains_manager_ownership_and_wire_order(
    width: int, dirty: bool
) -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, Register, Side, Signature
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    events = []
    wires = tuple(cirq.LineQubit.range(1, width + 1))

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            assert n == width and dim == 2
            events.append(("allocate", wires, False))
            return list(wires)

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            assert n == width and dim == 2
            events.append(("allocate", wires, True))
            return list(wires)

        def qfree(self, qubits: Iterable[cirq.LineQubit]) -> None:
            events.append(("free", tuple(qubits), None))

    dtype = QAny(width)
    builder, inputs = BloqBuilder.from_signature(
        Signature([Register("data", QAny(1)), Register("out", dtype, side=Side.RIGHT)])
    )
    out = builder.allocate(dtype=dtype, dirty=dirty)
    out = builder.join(builder.split(out)[::-1], dtype=dtype)
    composite = builder.finalize(data=inputs["data"], out=out)
    circuit, outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert tuple(circuit.all_operations()) == ()
    assert tuple(outputs["out"]) == wires[::-1]
    assert tuple(outputs["data"]) == (cirq.LineQubit(0),)
    assert events == [("allocate", wires, dirty)]
    assert get_cost_value(composite, QECGatesCost()).asdict() == {}
    assert get_cost_value(composite, QubitCount()) == width + 1


@pytest.mark.parametrize("dirty", [False, True])
def test_source_default_manager_rejects_releasing_unmanaged_caller_input(
    dirty: bool,
) -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, Register, Side, Signature

    builder, inputs = BloqBuilder.from_signature(
        Signature([Register("data", QAny(1), side=Side.LEFT)])
    )
    builder.free(inputs["data"], dirty=dirty)
    composite = builder.finalize()
    with pytest.raises(ValueError, match="was not allocated"):
        composite.to_cirq_circuit_and_quregs(
            qubit_manager=cirq.SimpleQubitManager(), data=np.array([cirq.LineQubit(0)])
        )
