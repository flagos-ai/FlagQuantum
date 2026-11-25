"""Reconcile the original S gate fixtures with the native interop boundaries."""

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.diagonal_gates_tutorial import diagonal_gate_examples
from flagquantum import Circuit
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.resource_generalization import PhaseCall
from flagquantum.core.ir import Instruction
from flagquantum.drawer.register_score_data import adjoint_wire_symbol
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.pennylane import (
    from_pennylane,
    to_pennylane,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_atomic_s_and_adjoint_leaf_and_display() -> None:
    from qualtran.bloqs.basic_gates.s_gate import _s_gate

    _, _, op, key = diagonal_gate_examples()[1]
    assert key == PhaseCall("s")
    forward = _s_gate.make()
    for inverse, source, code in (
        (False, forward, "s"),
        (True, forward.adjoint(), "sdg"),
    ):
        source_graph, leaves = source.call_graph()
        call = PhaseCall("s", adjoint=inverse)
        graph = symbolic_call_graph(call, {})
        assert list(source_graph.nodes) == [source] and not source_graph.edges
        assert not graph.direct_calls and graph.leaf_counts == {call: leaves[source]}
        symbol, attributes = (
            adjoint_wire_symbol("TextBox", {"text": "S"})
            if inverse
            else ("TextBox", {"text": "S"})
        )
        assert symbol == type(source.wire_symbol(source.signature[0])).__name__
        assert attributes["text"] == str(source)
        circuit = Circuit(1, dtype=torch.complex128)
        circuit.append(
            op if not inverse else FieldOperation(1, (Instruction(code, (0,)),))
        )
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(source.tensor_contract()),
            rtol=0,
            atol=1e-12,
        )


def test_original_h_s_sdagger_path_and_cirq_contract() -> None:
    import cirq
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import PlusState, SGate

    bb = BloqBuilder()
    q = bb.add(PlusState())
    q = bb.add(SGate(), q=q)
    q = bb.add(SGate().adjoint(), q=q)
    reference = bb.finalize(q=q).to_cirq_circuit()
    forward = Circuit(1, dtype=torch.complex128).h(0).s(0)
    assert to_cirq(forward.to_ir()) == cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))
    )
    assert (
        from_cirq(to_cirq(forward.to_ir())).instructions == forward.to_ir().instructions
    )

    forward.append(FieldOperation(1, (Instruction("sdg", (0,)),)))
    torch.testing.assert_close(
        forward.state().flatten(),
        torch.as_tensor(cirq.final_state_vector(reference, dtype=np.complex128)),
        rtol=0,
        atol=1e-12,
    )
    reference_on_native_wire = cirq.Circuit(reference).transform_qubits(
        {next(iter(reference.all_qubits())): cirq.LineQubit(0)}
    )
    assert to_cirq(forward.to_ir()) == reference_on_native_wire
    assert (
        from_cirq(to_cirq(forward.to_ir())).instructions == forward.to_ir().instructions
    )


def test_s_pennylane_forward_adapter_matches_original() -> None:
    qml = pytest.importorskip("pennylane")
    from qualtran.bloqs.basic_gates import SGate

    forward = Circuit(1, dtype=torch.complex128).s(0)
    script = to_pennylane(forward.to_ir())
    (gate,) = script.operations
    source = SGate().as_pl_op(wires=[0])
    assert gate == source
    assert from_pennylane(script).instructions == forward.to_ir().instructions
    torch.testing.assert_close(
        get_unitary(forward), torch.as_tensor(qml.matrix(gate)), rtol=0, atol=1e-12
    )
    inverse = Circuit(1).append(FieldOperation(1, (Instruction("sdg", (0,)),)))
    inverse_script = to_pennylane(inverse.to_ir())
    assert inverse_script.operations[0] == SGate().adjoint().as_pl_op(wires=[0])
    assert from_pennylane(inverse_script).instructions == inverse.to_ir().instructions
