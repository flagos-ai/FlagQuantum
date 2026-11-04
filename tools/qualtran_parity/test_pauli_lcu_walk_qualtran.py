"""Compare Pauli SELECT implementations inside the same qubitization walk."""

from collections import Counter
from importlib.metadata import version

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import arbitrary_state
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.compiler.deferred_operations import lower_select_pauli_lcu_to_gates
from flagquantum.simulation.unitary import get_unitary


def test_pauli_lcu_walk_qualtran() -> None:
    assert version("qualtran") == "0.7.0"
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU as Reference

    index = (0, 1, 2)
    prepare = arbitrary_state(
        torch.tensor([0.5, 0.3, 0.2, 0, 0, 0, 0, 0], dtype=torch.float64).sqrt()
    )
    reflection = fq.Circuit(3)
    reflection.x(0).x(1).x(2).h(2).ccx(0, 1, 2).h(2).x(2).x(1).x(0)
    reflection.x(0).z(0).x(0).z(0)
    circuit = fq.Circuit(4)
    circuit.append(
        SelectPauliLCU(3, 1, ("X", "-Y", "Z")).make_on(selection=index, target=(3,))
    )
    select = lower_select_pauli_lcu_to_gates(circuit.to_ir().instructions[0])
    encoding = fq.Circuit(select.n_qubits).compose(prepare, qubits=index)
    encoding.compose(select).compose(prepare.adjoint(), qubits=index)

    q = cirq.LineQubit.range(4)
    reference = Reference(
        3,
        1,
        tuple(
            cirq.DensePauliString(axis, coefficient=sign)
            for axis, sign in (("X", 1), ("Y", -1), ("Z", 1))
        ),
    )
    reference_select = cirq.Circuit(
        cirq.decompose(
            reference.on_registers(selection=q[:3], target=q[3:]),
            keep=lambda op: op.gate == cirq.TOFFOLI,
            on_stuck_raise=None,
        )
    )
    workspace = sorted(reference_select.all_qubits().difference(q))
    prep = get_unitary(prepare, dtype=torch.complex128).numpy()
    ref = cirq.Circuit(
        cirq.MatrixGate(prep).on(*q[:3]),
        reference_select,
        cirq.MatrixGate(prep.conj().T).on(*q[:3]),
    )
    stride = 2 ** len(workspace)
    simulator = cirq.Simulator(dtype=np.complex128, seed=42)
    columns = []
    for basis_state in range(16):
        result = simulator.simulate(
            ref, qubit_order=[*q, *workspace], initial_state=basis_state * stride
        ).final_state_vector.reshape(16, stride)
        np.testing.assert_allclose(result[:, 1:], 0, atol=1e-12)
        columns.append(result[:, 0])
    reference_block = np.column_stack(columns)
    # Check a coherent input as well, so basis-dependent branch phases cannot
    # silently pass the reconstructed-matrix checks.
    rng = np.random.default_rng(42)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    padded = np.zeros((16, stride), dtype=np.complex128)
    padded[:, 0] = state
    result = simulator.simulate(
        ref, qubit_order=[*q, *workspace], initial_state=padded.ravel()
    ).final_state_vector.reshape(16, stride)
    np.testing.assert_allclose(
        result[:, 0], reference_block @ state, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(result[:, 1:], 0, atol=1e-12)

    local_matrix = get_unitary(encoding, dtype=torch.complex128).numpy()
    local_stride = 2 ** (select.n_qubits - 4)
    np.testing.assert_allclose(
        local_matrix.reshape(16, local_stride, 16, local_stride)[:, 1:, :, 0],
        0,
        atol=1e-12,
    )
    h = np.array([[0.2, 0.5 + 0.3j], [0.5 - 0.3j, -0.2]])
    r = np.kron(get_unitary(reflection, dtype=torch.complex128).numpy(), np.eye(2))
    values, vectors = np.linalg.eigh(h)
    for block_encoding in (
        local_matrix[::local_stride, ::local_stride],
        reference_block,
    ):
        np.testing.assert_allclose(block_encoding[:2, :2], h, atol=1e-7, rtol=0)
        walk = r @ block_encoding
        for value, vector in zip(values, vectors.T, strict=True):
            flagged = np.pad(vector, (0, 14))
            partner = block_encoding @ flagged - value * flagged
            basis = np.column_stack((flagged, partner / np.linalg.norm(partner)))
            projected = basis.conj().T @ walk @ basis
            np.testing.assert_allclose(
                walk @ basis, basis @ projected, atol=1e-7, rtol=0
            )
            phase = np.arccos(value)
            np.testing.assert_allclose(
                np.sort(np.angle(np.linalg.eigvals(projected))),
                [-phase, phase],
                atol=1e-7,
                rtol=0,
            )

    for name, stage in (
        ("PREPARE", prepare),
        ("PREPARE dagger", prepare.adjoint()),
        ("reflection", reflection),
        ("FlagQuantum SELECT", select),
    ):
        print(name, Counter(gate.name for gate in stage.to_ir().instructions))
    print(
        "Qualtran SELECT",
        Counter(str(op.gate) for op in reference_select.all_operations()),
    )
    print("SELECT work qubits", select.n_qubits - 4, len(workspace))
