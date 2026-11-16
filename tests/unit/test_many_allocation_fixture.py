"""Compare allocation lifetime structure, phase and emitted primitives."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from tests.unit.many_allocation_fixture import AllocationMode, many_allocation_fixture
from tests.unit.register_source_graph import SourcePorts, SourceResult
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_t_leaf(
    bloq: Any, inputs: SourcePorts, desired: SourcePorts
) -> SourceResult | None:
    from qualtran.bloqs.basic_gates import TGate

    if not isinstance(bloq, TGate):
        return None
    assert not bloq.is_adjoint
    assert all(inputs[name] == wires for name, wires in desired.items())
    return (Instruction("t", inputs["q"]),), inputs


@pytest.mark.parametrize("mode", ["once", "many", "abstracted"])
@pytest.mark.parametrize("width", [1, 3, 10])
def test_many_allocation_schedules_use_one_slot(
    mode: AllocationMode, width: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.qubit_count_many_alloc import (
        TestManyAllocAbstracted,
        TestManyAllocMany,
        TestManyAllocOnce,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = {
        "once": TestManyAllocOnce,
        "many": TestManyAllocMany,
        "abstracted": TestManyAllocAbstracted,
    }[mode](width)
    record = many_allocation_fixture(width, mode)
    main = next(fn for fn in record.functions if fn.name == "main")
    assert main.registers is not None
    assert [(r.name, r.bitsize, r.shape) for r in main.registers] == [
        (r.name, r.bitsize, r.shape) for r in source.signature
    ]
    source_calls = source.decompose_bloq().bloq_instances
    original_nodes = Counter(type(node.bloq).__name__ for node in source_calls)
    if mode == "abstracted":
        assert original_nodes["_Inner"] == width
        assert sum(op.callee == "inner" for op in main.blocks[0].operations) == width
    else:
        assert (
            original_nodes["Allocate"]
            == original_nodes["Free"]
            == (1 if mode == "once" else width)
        )

    x = tuple(range(width - 1, -1, -1))
    clean = (width,)
    bound = bind_register_program(record, (x,), clean=clean)
    expected, events = source_workspace_gates(
        source, {"x": x}, clean, (), custom_leaf=source_t_leaf
    )
    assert_source(bound.circuit.instructions, expected)
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == list(
        events
    )
    assert bound.outputs == (x,)
    assert len(events) == 2 * (1 if mode == "once" else width)
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == width + 1
    cost = get_cost_value(source, QECGatesCost())
    assert (cost.t, cost.clifford, cost.measurement) == (width, 2 * width, 0)
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "cx": 2 * width,
        "t": width,
    }

    values = torch.arange(1, (1 << width) + 1, dtype=torch.float64)
    ket = torch.complex(values, 1 - values)
    ket /= ket.norm()
    axes = tuple(reversed(range(width)))
    vacuum = torch.tensor([1, 0], dtype=ket.dtype)
    initial = torch.kron(ket.reshape((2,) * width).permute(axes).flatten(), vacuum)
    source_tensor = torch.as_tensor(source.tensor_contract().copy(), dtype=ket.dtype)
    logical = source_tensor @ ket
    final = torch.kron(logical.reshape((2,) * width).permute(axes).flatten(), vacuum)
    circuit = Circuit(width + 1, inputs=initial, dtype=ket.dtype)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), final, atol=2e-14, rtol=0)

    for end in range(3, 3 * width + 1, 3):
        prefix = bound.circuit.instructions[:end]
        at_release = Circuit(width + 1, inputs=initial, dtype=ket.dtype)
        at_release.append(FieldOperation(width + 1, prefix))
        assert torch.count_nonzero(at_release.state().flatten()[1::2]) == 0
