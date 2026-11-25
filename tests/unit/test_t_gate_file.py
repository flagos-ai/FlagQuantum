"""Cover the locked T factory, magic-state construction and external boundaries."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import PhaseCall
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.pennylane import (
    from_pennylane,
    to_pennylane,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_t_factory_leaf_and_complex_matrices() -> None:
    from qualtran.bloqs.basic_gates.t_gate import _t_gate

    forward = _t_gate.make()
    signature = Signature.build(q=1)
    for inverse, source, name in (
        (False, forward, "t"),
        (True, forward.adjoint(), "tdg"),
    ):
        assert [(r.name, r.bitsize, r.side.value) for r in signature] == [
            (r.name, r.bitsize, r.side.value) for r in source.signature
        ]
        key = PhaseCall("t", adjoint=inverse)
        reference_graph, leaves = source.call_graph()
        graph = symbolic_call_graph(key, {})
        assert not reference_graph.edges and not graph.direct_calls
        assert graph.leaf_counts == {key: leaves[source]} == {key: 1}
        assert query_qec_gate_cost(key).t == source.t_complexity().t == 1
        circuit = Circuit(1, dtype=torch.complex128)
        circuit.append(FieldOperation(1, (Instruction(name, (0,)),)))
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(source.tensor_contract()),
            rtol=0,
            atol=2e-12,
        )


def test_original_magic_state_full_unitary_and_adjoint() -> None:
    from qualtran.bloqs.basic_gates.t_gate_test import TestTStateMaker

    source = TestTStateMaker()
    forward = FieldOperation(1, (Instruction("h", (0,)), Instruction("t", (0,))))
    circuit = Circuit(1, dtype=torch.complex128).append(forward)
    matrix = torch.as_tensor(source.tensor_contract())
    torch.testing.assert_close(get_unitary(circuit), matrix, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        circuit.state().flatten(), matrix[:, 0], rtol=0, atol=2e-12
    )
    inverse = circuit.adjoint()
    assert [gate.name for gate in inverse.to_ir().instructions] == ["tdg", "h"]
    torch.testing.assert_close(
        get_unitary(inverse),
        torch.as_tensor(source.adjoint().tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    circuit.append(FieldOperation(1, inverse.to_ir().instructions))
    torch.testing.assert_close(
        get_unitary(circuit), torch.eye(2, dtype=torch.complex128)
    )


def test_original_t_forward_bridges_and_adjoint_boundary() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import TGate

    qml = pytest.importorskip("pennylane")
    forward = Circuit(1, dtype=torch.complex128).t(0)
    ir = forward.to_ir()
    assert to_cirq(ir) == cirq.Circuit(cirq.T(cirq.LineQubit(0)))
    assert from_cirq(to_cirq(ir)).instructions == ir.instructions
    script = to_pennylane(ir)
    (op,) = script.operations
    assert op == TGate().as_pl_op(wires=[0])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        get_unitary(forward), torch.as_tensor(qml.matrix(op)), rtol=0, atol=2e-12
    )
    # The original Cirq fixture appends T-adjoint after a prepared T state.
    full = Circuit(1, dtype=torch.complex128).h(0).t(0)
    full.append(FieldOperation(1, (Instruction("tdg", (0,)),)))
    torch.testing.assert_close(
        get_unitary(full), get_unitary(Circuit(1, dtype=torch.complex128).h(0))
    )
    assert to_cirq(full.to_ir()) == cirq.Circuit(
        cirq.H(cirq.LineQubit(0)),
        cirq.T(cirq.LineQubit(0)),
        (cirq.T**-1)(cirq.LineQubit(0)),
    )
    assert from_cirq(to_cirq(full.to_ir())).instructions == full.to_ir().instructions
    assert (
        from_pennylane(to_pennylane(full.to_ir())).instructions
        == full.to_ir().instructions
    )
