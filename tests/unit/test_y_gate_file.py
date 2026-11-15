"""Check pinned Pauli Y and CY fixtures at native entry points."""

from dataclasses import asdict
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.primitives.basis_state import basis_state_ir
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.ecosystem.cirq import to_cirq
from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("bit", [0, 1])
def test_original_eigenstate_preparation_followed_by_y(bit: int) -> None:
    import cirq
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import MinusState, PlusState, YGate

    source = BloqBuilder()
    q = source.add((MinusState if bit else PlusState)())
    composite = source.finalize(q=source.add(YGate(), q=q))
    source_ops = cirq.decompose(
        composite.to_cirq_circuit(), keep=lambda op: op.gate in (cirq.X, cirq.H, cirq.Y)
    )
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(
        FieldOperation(1, (*basis_state_ir("x", bit, 0), *pauli_gate_ir("y", 0)))
    )
    exported = to_cirq(circuit.to_ir())
    assert [op.gate for op in exported.all_operations()] == [
        op.gate for op in source_ops
    ]
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.as_tensor(composite.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("opcode", ["y", "cy"])
def test_original_atomic_matrix_and_pennylane_roundtrip(
    opcode: Literal["y", "cy"],
) -> None:
    from qualtran.bloqs.basic_gates import CYGate, YGate
    from qualtran.cirq_interop.t_complexity_protocol import t_complexity
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    width = 1 if opcode == "y" else 2
    gates = (
        pauli_gate_ir("y", 0)
        if width == 1
        else pauli_gate_ir("y", 1, controls=(0,), values=(1,))
    )
    source = YGate() if width == 1 else YGate().controlled()
    assert isinstance(source, (YGate, CYGate))
    assert source.adjoint() == source
    assert [(register.name, register.bitsize) for register in source.signature] == (
        [("q", 1)] if width == 1 else [("ctrl", 1), ("target", 1)]
    )
    assert [(gate.name, gate.wires) for gate in gates] == [
        (opcode, tuple(range(width)))
    ]
    circuit = Circuit(width, dtype=torch.complex128)
    operation = FieldOperation(width, gates)
    circuit.append(operation)
    native_matrix = get_unitary(circuit)
    torch.testing.assert_close(
        native_matrix,
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
    exported = to_pennylane(circuit.to_ir())
    (op,) = exported.operations
    assert op == source.as_pl_op(wires=list(range(width)))
    assert from_pennylane(exported).instructions == circuit.to_ir().instructions
    torch.testing.assert_close(
        native_matrix, torch.as_tensor(op.matrix()), atol=2e-12, rtol=0
    )
    torch.testing.assert_close(
        get_unitary(circuit.append(operation)),
        torch.eye(1 << width, dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
    model = query_qec_gate_cost(CliffordCall(opcode, width))
    assert {name: value for name, value in asdict(model).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert t_complexity(source).clifford == model.clifford == 1


def test_original_active_cy_contracts_to_y() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CYGate, OneEffect, OneState

    source = BloqBuilder()
    target = source.add_register("q", 1)
    control, target = source.add(CYGate(), ctrl=source.add(OneState()), target=target)
    source.add(OneEffect(), q=control)
    reference = torch.as_tensor(
        source.finalize(q=target).tensor_contract(), dtype=torch.complex128
    )
    gates = (
        *basis_state_ir("z", 1, 0),
        *pauli_gate_ir("y", 1, controls=(0,), values=(1,)),
    )
    vectors = []
    for basis in torch.eye(4, dtype=torch.complex128)[0:2]:
        circuit = Circuit(2, inputs=basis, dtype=torch.complex128)
        circuit.append(FieldOperation(2, gates))
        network = build_tensor_network(circuit, dtype=torch.complex128)
        vectors.append(tensor_network_basis_effect(network, {0: ("z", 1)}).flatten())
    torch.testing.assert_close(
        torch.stack(vectors, dim=1), reference, atol=2e-12, rtol=0
    )


def test_original_cy_cirq_shape_operand_order_and_resource_key() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import CYGate

    from flagquantum.ecosystem.cirq.resource_keys import cirq_gate_resource_key

    source = CYGate().as_composite_bloq().to_cirq_circuit()
    (op,) = source.all_operations()
    assert op == cirq.Y(cirq.NamedQubit("target")).controlled_by(
        cirq.NamedQubit("ctrl")
    )
    assert cirq_gate_resource_key(op.gate) == CliffordCall("cy", 2)
    gates = pauli_gate_ir("y", 1, controls=(0,), values=(1,))
    assert tuple(gate.wires for gate in gates) == ((0, 1),)
    circuit = Circuit(2, dtype=torch.complex128).append(FieldOperation(2, gates))
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(cirq.unitary(op), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
