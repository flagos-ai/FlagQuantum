"""Verify executable THC rotations and report the Qualtran cost model."""

from collections import Counter
from importlib.metadata import version

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import THCRotations


def test_thc_rotations_complex_amplitudes_and_inverse() -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    assert version("qualtran") == "0.7.0"
    from qualtran.bloqs.chemistry.thc.select_bloq import THCRotations as CostModel
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    networks = (
        ((0, 1, 0.2), (1, 2, -0.4)),
        ((1, 2, 0.3),),
        ((0, 1, -0.1),),
    )
    # Use noncontiguous, reversed bindings and include a padded address.
    selection, target = (4, 0), (3, 1, 2)
    qubits = cirq.LineQubit.range(5)
    reference = cirq.Circuit()
    for address, network in enumerate(networks):
        values = divmod(address, 2)
        for left, right, angle in network:
            reference.append(
                cirq.givens(angle)
                .on(qubits[target[left]], qubits[target[right]])
                .controlled_by(*(qubits[q] for q in selection), control_values=values)
            )

    rng = np.random.default_rng(29)
    state = rng.normal(size=32) + 1j * rng.normal(size=32)
    state /= np.linalg.norm(state)
    circuit = fq.Circuit(5, dtype=torch.complex128, inputs=torch.from_numpy(state))
    operation = THCRotations.make_on(
        selection=selection, target=target, networks=networks
    )
    circuit.append(operation)
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(reference, qubit_order=qubits, initial_state=state)
        .final_state_vector
    )
    np.testing.assert_allclose(
        circuit.state().detach().numpy().reshape(-1), expected, rtol=0, atol=1e-12
    )
    gate_ir = circuit.to_ir()
    assert gate_ir.n_wires == len(qubits)
    assert all(item.matrix is None for item in gate_ir.instructions)
    print(
        "FlagQuantum exact gates:",
        dict(Counter(item.name for item in gate_ir.instructions)),
    )

    circuit.append(operation.adjoint())
    torch.testing.assert_close(
        circuit.state().flatten(), torch.from_numpy(state), rtol=0, atol=1e-12
    )

    # Qualtran's THCRotations is an analytical estimate for a quantized
    # QROM/phase-gradient implementation. It has no executable decomposition.
    model = CostModel(num_mu=2, num_spin_orb=6, num_bits_theta=8)
    print("Qualtran modeled forward resources:", get_cost_value(model, QECGatesCost()))
    print(
        "Qualtran modeled adjoint resources:",
        get_cost_value(model.adjoint(), QECGatesCost()),
    )
