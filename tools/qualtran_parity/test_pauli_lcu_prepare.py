"""Compare native PREPARE and the complete signed Pauli block encoding."""

from collections import Counter
from importlib.metadata import version

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.lcu_prepare import prepare_lcu
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.simulation.unitary import get_unitary


def test_pauli_lcu_prepare() -> None:
    assert version("qualtran") == "0.7.0"
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU as Reference
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
    from qualtran.cirq_interop.testing import GateHelper

    # Dyadic weights let both implementations encode the same operator exactly.
    weights = (1.0, -0.5, 0.25, 0.25)
    prepare, alpha = prepare_lcu(weights)
    assert alpha == 2
    local = fq.Circuit(3, dtype=torch.complex128).compose(prepare, qubits=(0, 1))
    local.append(
        SelectPauliLCU(2, 1, ("X", "-Y", "Z", "I")).make_on(
            selection=(0, 1), target=(2,)
        )
    )
    local.compose(prepare.adjoint(), qubits=(0, 1))
    # Exercise the stable run entry point as well as extracting the full block.
    fq.run(local)
    block = get_unitary(local, dtype=torch.complex128).numpy()[:2, :2]

    alias = StatePreparationAliasSampling.from_probabilities(
        [abs(weight) for weight in weights], precision=0.125
    )
    select = Reference(
        2,
        1,
        tuple(
            cirq.DensePauliString(axis, coefficient=1 if weight > 0 else -1)
            for axis, weight in zip("XYZI", weights, strict=True)
        ),
    )
    encoding = LCUBlockEncoding(select, alias)

    def decompose(bloq):
        helper = GateHelper(bloq)
        manager = cirq.GreedyQubitManager(prefix="work", maximize_reuse=True)
        circuit = cirq.Circuit(
            cirq.decompose(helper.operation, context=cirq.DecompositionContext(manager))
        )
        return circuit, helper.quregs

    reference_prepare, registers = decompose(alias)
    index = list(registers["selection"])
    work = sorted(reference_prepare.all_qubits().difference(index))
    simulator = cirq.Simulator(dtype=np.complex128, seed=42)
    expected_probabilities = np.abs(weights) / alpha
    reference_state = simulator.simulate(
        reference_prepare, qubit_order=[*index, *work]
    ).final_state_vector.reshape(4, -1)
    np.testing.assert_allclose(
        np.sum(abs(reference_state) ** 2, axis=1), expected_probabilities, atol=1e-12
    )
    np.testing.assert_allclose(
        abs(prepare.state().numpy().ravel()) ** 2, expected_probabilities, atol=1e-7
    )

    reference_encoding, registers = decompose(encoding)
    data = list(registers["target"])
    remaining = sorted(reference_encoding.all_qubits().difference(data))
    columns = []
    for bit in range(2):
        result = simulator.simulate(
            reference_encoding,
            qubit_order=[*data, *remaining],
            initial_state=bit * (1 << len(remaining)),
        ).final_state_vector.reshape(2, -1)
        columns.append(result[:, 0])
    expected = np.array([[0.25, 0.5 + 0.25j], [0.5 - 0.25j, 0.0]])
    np.testing.assert_allclose(np.column_stack(columns), expected, atol=1e-12)
    np.testing.assert_allclose(block, expected, atol=1e-7)

    print(
        "FlagQuantum PREPARE",
        prepare.n_qubits,
        Counter(gate.name for gate in prepare.to_ir().instructions),
    )
    print(
        "Qualtran PREPARE",
        len(reference_prepare.all_qubits()),
        Counter(
            (
                "feedback"
                if isinstance(op, cirq.ClassicallyControlledOperation)
                else (
                    "measurement"
                    if cirq.is_measurement(op)
                    else (
                        "reset"
                        if isinstance(op.gate, cirq.ResetChannel)
                        else f"{len(op.qubits)}q"
                    )
                )
            )
            for op in reference_prepare.all_operations()
        ),
    )
    print("Qualtran block encoding qubits", len(reference_encoding.all_qubits()))
