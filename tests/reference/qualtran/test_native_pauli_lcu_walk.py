"""Check the native Qualtran walk against the FlagQuantum Pauli LCU example."""

from importlib.metadata import version

import pytest

from examples.algorithms.pauli_lcu_block_encoding import main as verify_flagquantum


def test_native_pauli_lcu_walk() -> None:
    assert version("qualtran") == "0.7.0"
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
    from qualtran.cirq_interop.testing import GateHelper

    verify_flagquantum()
    # Use the native minimum-width address register for three terms.
    precision = 0.17
    prepare = StatePreparationAliasSampling.from_probabilities(
        [1.0, 0.6, 0.4], precision=precision
    )
    select = SelectPauliLCU(
        2,
        1,
        tuple(
            cirq.DensePauliString(axis, coefficient=sign)
            for axis, sign in (("X", 1), ("Y", -1), ("Z", 1))
        ),
    )
    encoding = LCUBlockEncoding(select, prepare)
    walk = QubitizationWalkOperator(encoding)
    assert float(encoding.alpha) == 2
    helper = GateHelper(walk)
    data = tuple(helper.quregs["target"])

    def decompose(gate):
        context = cirq.DecompositionContext(
            cirq.GreedyQubitManager(prefix="work", maximize_reuse=True)
        )
        return cirq.Circuit(
            cirq.decompose(gate.on_registers(**helper.quregs), context=context)
        )

    block_circuit = decompose(encoding)
    upstream_walk = decompose(walk)
    # LCUBlockEncoding.signal_state omits the alias junk registers in v0.7.0.
    # Reflect about zero on the entire signal register for the comparison.
    signal = tuple(
        q
        for reg in encoding.signature
        if reg.name != "target"
        for q in helper.quregs[reg.name].flat
    )
    reflection = ReflectionUsingPrepare.reflection_around_zero(
        [len(signal)], global_phase=-1
    )
    reflection_helper = GateHelper(reflection)
    reflection_qubits = [
        q for register in reflection_helper.quregs.values() for q in register.flat
    ]
    context = cirq.DecompositionContext(
        cirq.GreedyQubitManager(prefix="reflect", maximize_reuse=True)
    )
    reflection_circuit = cirq.Circuit(
        cirq.decompose(reflection_helper.operation, context=context)
    )
    mapping = dict(zip(reflection_qubits, signal, strict=True))
    reflection_circuit = reflection_circuit.transform_qubits(
        lambda q: mapping.get(q, q)
    )
    walk_circuit = block_circuit + reflection_circuit

    simulator = cirq.Simulator(dtype=np.complex128, seed=42)

    def simulate(program, vector):
        rest = sorted(program.all_qubits().difference(data))
        stride = 2 ** len(rest)
        initial = np.zeros((2, stride), dtype=np.complex128)
        initial[:, 0] = vector
        return simulator.simulate(
            program, qubit_order=[*data, *rest], initial_state=initial.ravel()
        ).final_state_vector.reshape(2, stride)

    block = np.column_stack(
        [simulate(block_circuit, vector)[:, 0] for vector in np.eye(2)]
    )
    expected = np.array([[0.2, 0.5 + 0.3j], [0.5 - 0.3j, -0.2]])
    np.testing.assert_allclose(block, block.conj().T, atol=1e-12, rtol=0)
    # At most three coefficient errors of size precision, divided by alpha.
    assert np.linalg.norm(block - expected, ord=2) <= 3 * precision / encoding.alpha
    values, vectors = np.linalg.eigh(block)
    for value, vector in zip(values, vectors.T, strict=True):
        # Spectral moments of the zero-signal eigenvector are cos(k * phase).
        for power in (1, 2, 3):
            observed = np.vdot(vector, simulate(walk_circuit * power, vector)[:, 0])
            np.testing.assert_allclose(
                observed, np.cos(power * np.arccos(value)), atol=1e-10, rtol=0
            )

    native_second_moment = np.vdot(
        vectors[:, 0], simulate(upstream_walk * 2, vectors[:, 0])[:, 0]
    )
    print(
        "Native walk second-moment error:",
        abs(native_second_moment - (2 * values[0] ** 2 - 1)),
    )
    print(
        "Qualtran block error (operator norm):", np.linalg.norm(block - expected, ord=2)
    )
    print("Qualtran walk qubits:", len(walk_circuit.all_qubits()))
    print("Qualtran emitted operations:", sum(1 for _ in walk_circuit.all_operations()))
