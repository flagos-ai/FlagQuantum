"""Exercise deferred Pauli table selection on remapped registers."""

from importlib.metadata import version

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU


@pytest.mark.parametrize("control_val", [None, 0, 1])
def test_select_pauli_lcu_qualtran_parity(control_val: int | None) -> None:
    assert version("qualtran") == "0.7.0", "Install qualtran==0.7.0 for parity tests."
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU as Reference

    words = ("XI", "-YZ", "IX", "ZY")
    control_width = int(control_val is not None)
    width = 4 + control_width
    order = (width - 1, *range(width - 1))
    qubits = cirq.LineQubit.range(width)
    operands = [qubits[index] for index in order]
    reference = Reference(
        2,
        2,
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in words
        ),
        control_val=control_val,
    )
    bindings = {
        "selection": operands[control_width : control_width + 2],
        "target": operands[-2:],
    }
    if control_width:
        bindings["control"] = operands[:1]
    decomposition = cirq.Circuit(cirq.decompose(reference.on_registers(**bindings)))
    ancillas = sorted(decomposition.all_qubits().difference(qubits))
    rng = np.random.default_rng(42)
    state = rng.normal(size=2**width) + 1j * rng.normal(size=2**width)
    state /= np.linalg.norm(state)
    initial = np.zeros((state.size, 2 ** len(ancillas)), dtype=np.complex128)
    initial[:, 0] = state
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            decomposition,
            qubit_order=[*qubits, *ancillas],
            initial_state=initial.ravel(),
        )
        .final_state_vector.reshape(initial.shape)
    )
    np.testing.assert_allclose(expected[:, 1:], 0, atol=1e-12)

    local = SelectPauliLCU(2, 2, words, control_val).append_to(
        fq.Circuit(width),
        control=tuple(range(control_width)),
        selection=(control_width, control_width + 1),
        target=(width - 2, width - 1),
    )
    circuit = fq.Circuit(width, dtype=torch.complex128, inputs=torch.from_numpy(state))
    circuit.compose(local, qubits=order)
    np.testing.assert_allclose(
        circuit.state().numpy().reshape(-1), expected[:, 0], atol=1e-12, rtol=0
    )
