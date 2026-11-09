"""Check the physical lowering and source resource boundaries for discarding."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import ZeroCostCall
from flagquantum.compiler.registers.primitives.measurement import discard_qubit_ir
from flagquantum.core.ir import Instruction
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_discard_returns_slot_to_zero_without_a_classical_result(wire: int) -> None:
    values = torch.arange(1, 9, dtype=torch.float64)
    state = torch.complex(values, values.remainder(5) - 2)
    state /= state.norm()
    gates = discard_qubit_ir(wire)
    assert gates == (Instruction("reset", (wire,), metadata={"is_dynamic": True}),)
    assert Counter(gate.name for gate in gates) == {"reset": 1}
    circuit = DynamicCircuit(3, inputs=state, dtype=state.dtype)
    circuit.append(FieldOperation(3, gates))
    expected = state.reshape(2, 2, 2).movedim(wire, 0).reshape(2, 4)
    density = circuit.density_matrix().reshape(2, 2, 2, 2, 2, 2)
    density = density.movedim((wire, wire + 3), (0, 3)).reshape(2, 4, 2, 4)
    torch.testing.assert_close(density[0, :, 0], expected.T @ expected.conj())
    assert torch.count_nonzero(density[1]) == 0
    assert torch.count_nonzero(density[:, :, 1]) == 0
    result = run_dynamic(circuit, shots=32, seed=13, strategy="trajectory")
    assert result.classical_bits.shape[-1] == 0
    raw = result.final_states.reshape(32, 2, 2, 2).movedim(wire + 1, 1)
    assert torch.count_nonzero(raw[:, 1]) == 0


@pytest.mark.ftqc_reference
def test_discard_resource_contract_distinguishes_classical_and_quantum_inputs() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CBit, DecomposeNotImplementedError, QBit, Side
    from qualtran.bloqs.basic_gates import Discard, DiscardQ
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    classical, quantum = Discard(), DiscardQ()
    assert tuple((r.side, r.dtype) for r in classical.signature) == (
        (Side.LEFT, CBit()),
    )
    assert tuple((r.side, r.dtype) for r in quantum.signature) == ((Side.LEFT, QBit()),)
    assert classical.call_classically(c=0) == classical.call_classically(c=1) == ()
    assert get_cost_value(classical, QubitCount()) == 0
    assert get_cost_value(classical, QECGatesCost()).asdict() == {}
    assert query_qec_gate_cost(ZeroCostCall("discard")) == QECGateCost()
    assert get_cost_value(quantum, QubitCount()) == 1
    with pytest.raises(
        DecomposeNotImplementedError, match="does not declare a decomposition"
    ):
        get_cost_value(quantum, QECGatesCost())


def test_discard_validates_wire() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        discard_qubit_ir(-1)
