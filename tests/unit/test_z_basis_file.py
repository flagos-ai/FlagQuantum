"""Reconcile the remaining pinned Z-basis fixtures with native entry points."""

from collections import Counter
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.layout.int_vector import int_state_ir
from flagquantum.compiler.registers.primitives.basis_state import basis_state_ir
from flagquantum.compiler.registers.primitives.measurement import measure_z_clear_ir
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
    tensor_network_basis_measurement,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("bit", [0, 1])
def test_original_z_preparation_followed_by_gate_and_matching_effect(
    bit: int,
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import (
        OneEffect,
        OneState,
        ZeroEffect,
        ZeroState,
        ZGate,
    )

    state_type = OneState if bit else ZeroState
    effect_type = OneEffect if bit else ZeroEffect
    source = BloqBuilder()
    q = source.add(state_type())
    composite = source.finalize(q=source.add(ZGate(), q=q))
    original = list(composite.to_cirq_circuit().all_operations())

    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, basis_state_ir("z", bit, 0)))
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    overlap = tensor_network_basis_effect(plan, {0: ("z", bit)})
    projector = BloqBuilder()
    projector.add(effect_type(), q=projector.add(state_type()))
    torch.testing.assert_close(
        overlap.reshape(()),
        torch.as_tensor(projector.finalize().tensor_contract()).reshape(()),
        atol=2e-12,
        rtol=0,
    )
    circuit.append(FieldOperation(1, pauli_gate_ir("z", 0)))
    exported = to_cirq(circuit.to_ir())
    assert [op.gate for op in exported.all_operations()] == [op.gate for op in original]
    assert from_cirq(exported).instructions == circuit.to_ir().instructions
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.as_tensor(composite.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_original_z_and_cz_foreign_atomic_roundtrips(controlled: bool) -> None:
    from qualtran.bloqs.basic_gates import CZ, ZGate

    width = 2 if controlled else 1
    source = CZ() if controlled else ZGate()
    gates = (
        pauli_gate_ir("z", 1, controls=(0,), values=(1,))
        if controlled
        else pauli_gate_ir("z", 0)
    )
    circuit = Circuit(width, dtype=torch.complex128).append(
        FieldOperation(width, gates)
    )
    cirq_program = to_cirq(circuit.to_ir())
    (op,) = cirq_program.all_operations()
    (source_op,) = source.as_composite_bloq().to_cirq_circuit().all_operations()
    assert op.gate == source_op.gate
    assert from_cirq(cirq_program).instructions == gates
    pennylane_program = to_pennylane(circuit.to_ir())
    assert from_pennylane(pennylane_program).instructions == gates
    if not controlled:
        assert pennylane_program.operations[0] == source.as_pl_op(wires=[0])
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("basis,bit", [("z", 0), ("z", 1), ("x", 0), ("x", 1)])
def test_original_z_measurement_classical_output_density(
    basis: Literal["x", "z"], bit: int
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import (
        MeasureZ,
        MinusState,
        OneState,
        PlusState,
        ZeroState,
    )

    states = {"x": (PlusState, MinusState), "z": (ZeroState, OneState)}
    source = BloqBuilder()
    c = source.add(MeasureZ(), q=source.add(states[basis][bit]()))
    expected = source.finalize(c=c).tensor_contract(superoperator=True)
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, basis_state_ir(basis, bit, 0)))
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    branches = tensor_network_basis_measurement(plan, 0, basis="z")
    weights = torch.stack([branch.abs().square().sum() for branch in branches])
    torch.testing.assert_close(
        torch.diag(weights).to(torch.complex128),
        torch.as_tensor(expected),
        atol=2e-12,
        rtol=0,
    )


def test_original_z_measurement_word_order_and_physical_cleanup() -> None:
    from qualtran import BloqBuilder, QUInt
    from qualtran.bloqs.basic_gates import IntState, MeasureZ

    source = BloqBuilder()
    wire_values = source.split(source.add(IntState(val=52, bitsize=8)))
    for index in range(8):
        wire_values[index] = source.add(MeasureZ(), q=wire_values[index])
    (word,) = source.finalize(outs=wire_values).call_classically()
    assert list(word) == list(QUInt(8).to_bits(52))

    wires = tuple(reversed(range(8)))
    preparation = int_state_ir(wires, 52)
    measurement = tuple(
        gate
        for slot, wire in enumerate(wires)
        for gate in measure_z_clear_ir(wire, slot)
    )
    assert Counter(gate.name for gate in preparation) == {"x": 3}
    assert Counter(gate.name for gate in measurement) == {"measure": 8, "x": 8}
    circuit = DynamicCircuit(8, dtype=torch.complex128)
    circuit.append(FieldOperation(8, preparation))
    circuit.append(FieldOperation(8, measurement))
    result = run_dynamic(circuit, shots=1, seed=7, strategy="trajectory")
    assert result.classical_bits.reshape(-1).tolist() == list(word)
    expected = torch.zeros(256, dtype=torch.complex128)
    expected[0] = 1
    torch.testing.assert_close(
        result.final_states.reshape(-1), expected, atol=2e-12, rtol=0
    )
