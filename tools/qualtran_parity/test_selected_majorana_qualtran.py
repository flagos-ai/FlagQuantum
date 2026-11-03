"""Optional parity tests against the Qualtran 0.7.0 release."""

from importlib.metadata import version

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.multiplexers import SelectedMajoranaFermion


@pytest.mark.parametrize("target_gate", ["X", "Y", "Z"])
def test_qualtran_coherent_majorana_parity(target_gate: str) -> None:
    assert version("qualtran") == "0.7.0", "Install qualtran==0.7.0 for parity tests."
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion as QualtranMajorana,
    )

    qubits = cirq.LineQubit.range(7)
    operation = QualtranMajorana.make_on(
        control=qubits[:1],
        selection=qubits[1:3],
        target=qubits[3:],
        target_gate=getattr(cirq, target_gate),
    )
    decomposition = cirq.Circuit(cirq.decompose(operation))
    ancillas = sorted(decomposition.all_qubits().difference(qubits))

    rng = np.random.default_rng(42)
    state = rng.normal(size=128) + 1j * rng.normal(size=128)
    state /= np.linalg.norm(state)
    ancilla_zero = np.zeros(2 ** len(ancillas), dtype=np.complex128)
    ancilla_zero[0] = 1
    reference = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            decomposition,
            qubit_order=[*qubits, *ancillas],
            initial_state=np.kron(state, ancilla_zero),
        )
        .final_state_vector.reshape(128, -1)
    )
    np.testing.assert_allclose(reference[:, 1:], 0, atol=1e-12)

    circuit = Circuit(7, dtype=torch.complex128, inputs=torch.from_numpy(state))
    SelectedMajoranaFermion.make_on(
        control=(0,), selection=(1, 2), target=(3, 4, 5, 6), target_gate=target_gate
    ).append_to(circuit)
    np.testing.assert_allclose(
        circuit.state().detach().numpy().reshape(-1),
        reference[:, 0],
        atol=1e-12,
        rtol=0,
    )
