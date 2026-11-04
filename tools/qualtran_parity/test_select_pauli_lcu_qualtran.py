"""Exercise deferred Pauli table selection on remapped registers."""

from collections import Counter
from importlib.metadata import version

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.compiler.deferred_operations import lower_select_pauli_lcu_to_gates


@pytest.mark.parametrize(
    ("selection_width", "control_val"),
    [(2, None), (2, 0), (2, 1), (3, None), (4, None)],
)
def test_select_pauli_lcu_qualtran_parity(
    selection_width: int, control_val: int | None
) -> None:
    assert version("qualtran") == "0.7.0", "Install qualtran==0.7.0 for parity tests."
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU as Reference

    words = ("XI", "-YZ", "IX", "ZY") * (2 ** (selection_width - 2))
    control_width = int(control_val is not None)
    width = selection_width + 2 + control_width
    order = (width - 1, *range(width - 1))
    qubits = cirq.LineQubit.range(width)
    operands = [qubits[index] for index in order]
    reference = Reference(
        selection_width,
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
        "selection": operands[control_width : control_width + selection_width],
        "target": operands[-2:],
    }
    if control_width:
        bindings["control"] = operands[:1]
    context = cirq.DecompositionContext(
        cirq.GreedyQubitManager(prefix="work", maximize_reuse=True)
    )
    decomposition = cirq.Circuit(
        cirq.decompose(reference.on_registers(**bindings), context=context)
    )
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

    circuit = fq.Circuit(width, dtype=torch.complex128, inputs=torch.from_numpy(state))
    op = SelectPauliLCU(selection_width, 2, words, control_val).make_on(
        control=order[:control_width],
        selection=order[control_width:-2],
        target=order[-2:],
    )
    circuit.append(op)
    np.testing.assert_allclose(
        circuit.state().numpy().reshape(-1), expected[:, 0], atol=1e-12, rtol=0
    )

    instruction = circuit.to_ir().instructions[0]
    lowered = lower_select_pauli_lcu_to_gates(instruction)
    workspace = tuple(range(width, lowered.n_qubits))
    padded = np.zeros((state.size, 2 ** len(workspace)), dtype=np.complex128)
    padded[:, 0] = state
    gates = fq.Circuit(
        lowered.n_qubits,
        dtype=torch.complex128,
        inputs=torch.from_numpy(padded.ravel()),
    )
    gates.compose(lowered, qubits=(*instruction.wires, *workspace))
    observed = gates.state().numpy().reshape(padded.shape)
    np.testing.assert_allclose(observed[:, 0], expected[:, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(observed[:, 1:], 0, atol=1e-12, rtol=0)

    # Report emitted operations by arity. Keep feedback, measurement, and
    # reset separate rather than interpreting them as unitary gates.
    reference_counts: Counter[str] = Counter()
    for operation in decomposition.all_operations():
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            category = "feedback"
        elif cirq.is_measurement(operation):
            category = "measurement"
        elif isinstance(operation.gate, cirq.ResetChannel):
            category = "reset"
        else:
            category = f"{len(operation.qubits)}q"
        reference_counts[category] += 1
    local_counts = Counter(
        f"{len(gate.wires)}q" for gate in lowered.to_ir().instructions
    )
    print(
        f"entries={len(words)} control={control_val}: "
        f"FlagQuantum work={len(workspace)} {dict(local_counts)}; "
        f"Qualtran work={len(ancillas)} {dict(reference_counts)}"
    )
