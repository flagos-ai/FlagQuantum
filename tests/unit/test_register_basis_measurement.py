"""Verify measured physical slot reuse against the source basis instruments."""

from collections import Counter
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    MeasurementCall,
)
from flagquantum.compiler.registers.measurement import (
    measure_x_clear_ir,
    measure_z_clear_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("basis", ["x", "z"])
def test_consuming_measurement_resources_and_all_raw_branches(
    basis: Literal["x", "z"],
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CBit, QBit, Side
    from qualtran.bloqs.basic_gates import (
        MeasureX,
        MeasureZ,
        MinusEffect,
        OneEffect,
        PlusEffect,
        ZeroEffect,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    wire, slot = 1, 3
    if basis == "x":
        source, effects, gates = (
            MeasureX(),
            (PlusEffect, MinusEffect),
            measure_x_clear_ir(wire, slot),
        )
        prefix = (Instruction("h", (wire,)),)
    else:
        source, effects, gates = (
            MeasureZ(),
            (ZeroEffect, OneEffect),
            measure_z_clear_ir(wire, slot),
        )
        prefix = ()
    q, c = source.signature
    assert (q.side, q.dtype, c.side, c.dtype) == (Side.LEFT, QBit(), Side.RIGHT, CBit())
    assert get_cost_value(source, QubitCount()) == 1
    assert get_cost_value(source, QECGatesCost()).asdict() == {"measurement": 1}
    cost = query_qec_gate_cost(MeasurementCall(basis))
    assert cost.measurement == 1
    assert cost.t == cost.clifford == cost.rotation == 0
    assert gates == (
        *prefix,
        Instruction(
            "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": slot}
        ),
        Instruction("x", (wire,), metadata={"conditions": ((slot, 1),)}),
    )
    assert Counter(gate.name for gate in gates) == Counter(
        {
            "h": len(prefix),
            "measure": 1,
            "x": 1,
        }
    )

    state = torch.tensor(
        [complex(i + 1, (3 * i) % 7 - 2) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    for bit, effect in enumerate(effects):
        bra = torch.as_tensor(effect().tensor_contract(), dtype=state.dtype).reshape(2)
        expected = torch.tensordot(state.reshape(2, 2, 2), bra, dims=([1], [0]))
        projected = forced_branch(state, gates, (bit,)).reshape(2, 2, 2)
        torch.testing.assert_close(projected[:, 0], expected, atol=2e-12, rtol=0)
        assert torch.count_nonzero(projected[:, 1]) == 0


@pytest.mark.parametrize("basis", ["x", "z"])
def test_dynamic_measurement_writes_declared_classical_slot_and_clears_wire(
    basis: Literal["x", "z"],
) -> None:
    circuit = DynamicCircuit(2, dtype=torch.complex128)
    circuit.h(0).s(0).x(1)
    if basis == "x":
        circuit.h(1)
    emit = measure_x_clear_ir if basis == "x" else measure_z_clear_ir
    circuit.append(FieldOperation(2, emit(1, 3)))
    result = run_dynamic(circuit, shots=8, seed=9, strategy="trajectory")
    assert torch.all(result.classical_bits[:, 3] == 1)
    expected = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128)
    expected /= expected.norm()
    torch.testing.assert_close(result.final_states.reshape(8, 4), expected.expand(8, 4))
