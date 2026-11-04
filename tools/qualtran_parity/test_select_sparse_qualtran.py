"""Complex amplitude parity for the sparse chemistry SELECT oracle."""

from importlib.metadata import version

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.multiplexers import SelectSparse
from flagquantum.compiler.select_lowering import lower_select_program
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tools.qualtran_parity.clifford_t_counts import clifford_t_counts


def test_select_sparse_qualtran() -> None:
    num_spin_orb = 4
    assert version("qualtran") == "0.7.0"
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse as Reference

    width = (num_spin_orb // 2 - 1).bit_length()
    orbitals = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(4))
    alpha, beta, flag = range(4 * width, 4 * width + 3)
    n_qubits = flag + 1 + num_spin_orb
    qubits = cirq.LineQubit.range(n_qubits)
    op = Reference(num_spin_orb).on_registers(
        **dict(
            zip(
                ("p", "q", "r", "s"),
                (qubits[i * width : (i + 1) * width] for i in range(4)),
                strict=True,
            )
        ),
        alpha=[qubits[alpha]],
        beta=[qubits[beta]],
        flag_1b=[qubits[flag]],
        sys=qubits[flag + 1 :],
    )
    context = cirq.DecompositionContext(
        cirq.GreedyQubitManager(prefix="work", maximize_reuse=True)
    )
    decomposition = cirq.Circuit(cirq.decompose(op, context=context))
    ancillas = sorted(decomposition.all_qubits().difference(qubits))
    rng = np.random.default_rng(42)
    state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    state /= np.linalg.norm(state)
    zero = np.zeros(2 ** len(ancillas), dtype=np.complex128)
    zero[0] = 1
    reference = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            decomposition,
            qubit_order=[*qubits, *ancillas],
            initial_state=np.kron(state, zero),
        )
        .final_state_vector.reshape(2**n_qubits, -1)
    )
    np.testing.assert_allclose(reference[:, 1:], 0, atol=1e-12)

    p, q, r, s = orbitals
    circuit = Circuit(n_qubits, dtype=torch.complex128, inputs=torch.from_numpy(state))
    circuit.append(
        SelectSparse.make_on(
            p=p,
            q=q,
            r=r,
            s=s,
            alpha=alpha,
            beta=beta,
            flag_1b=flag,
            sys=range(flag + 1, n_qubits),
        )
    )
    logical_ops = list(cirq.flatten_to_ops(cirq.decompose_once(op)))
    instructions = circuit.to_ir().instructions
    assert [instruction.name for instruction in instructions] == [
        "s",
        "selected_majorana_fermion",
        "selected_majorana_fermion",
        "selected_majorana_fermion",
        "selected_majorana_fermion",
    ]
    assert all(instruction.matrix is None for instruction in instructions)
    assert len(logical_ops) == len(instructions)
    lowered = lower_select_program(circuit.to_ir(), "unitary_ladder")
    local_workspace = lowered.n_wires - n_qubits
    assert local_workspace == 1
    assert all(instruction.matrix is None for instruction in lowered.instructions)
    padded_state = np.zeros((state.size, 1 << local_workspace), dtype=np.complex128)
    padded_state[:, 0] = state
    local = Circuit.from_ir(lowered, inputs=torch.from_numpy(padded_state.ravel()))
    actual = local.state().detach().numpy().reshape(state.size, -1)
    np.testing.assert_allclose(actual[:, 1:], 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual[:, 0], reference[:, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        actual[:, 0], circuit.state().detach().numpy().reshape(-1), atol=1e-12, rtol=0
    )

    dynamic_ir = lower_select_program(circuit.to_ir(), "dynamic_unary_iteration")
    workspace = dynamic_ir.n_wires - n_qubits
    assert workspace == len(ancillas) == 3
    padded_state = np.zeros((state.size, 1 << workspace), dtype=np.complex128)
    padded_state[:, 0] = state
    dynamic = DynamicCircuit.from_ir(
        dynamic_ir, inputs=torch.from_numpy(padded_state.ravel())
    )
    trajectories = run_dynamic(dynamic, shots=8, seed=31).final_states
    for trajectory in trajectories.detach().numpy().reshape(8, state.size, -1):
        np.testing.assert_allclose(trajectory[:, 1:], 0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            trajectory[:, 0], reference[:, 0], atol=1e-12, rtol=0
        )

    gate_map = {
        "x": cirq.X,
        "h": cirq.H,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "cx": cirq.CNOT,
        "ccx": cirq.TOFFOLI,
    }
    local_qubits = cirq.LineQubit.range(lowered.n_wires)
    local_operations = [
        gate_map[item.name].on(*(local_qubits[q] for q in item.wires))
        for item in lowered.instructions
    ]
    unitary_cost = clifford_t_counts(cirq, cirq.decompose(local_operations))
    gate_map.update(
        {
            "cy": cirq.Y.controlled(),
            "cz": cirq.CZ,
            "t": cirq.T,
            "tdg": cirq.T**-1,
            "reset": cirq.ResetChannel(),
        }
    )
    local_qubits = cirq.LineQubit.range(dynamic_ir.n_wires)
    operations = []
    for item in dynamic_ir.instructions:
        wires = tuple(local_qubits[q] for q in item.wires)
        if item.name == "measure":
            operation = cirq.measure(*wires, key=str(item.metadata["classical_bit"]))
        else:
            operation = gate_map[item.name].on(*wires)
            if "conditions" in item.metadata:
                ((bit, value),) = item.metadata["conditions"]
                assert value == 1
                operation = operation.with_classical_controls(str(bit))
        operations.append(operation)
    dynamic_cost = clifford_t_counts(cirq, cirq.decompose(operations))
    # Qualtran exposes some Clifford gates as atomic bloq wrappers. Translate
    # those wrappers before the shared Cirq decomposition so a CNOT costs the
    # same in both columns.
    reference_cliffords = {CNOT(): cirq.CNOT, XGate(): cirq.X}
    reference_operations = []
    for operation in decomposition.all_operations():
        equivalent = reference_cliffords.get(getattr(operation.gate, "bloq", None))
        reference_operations.append(
            operation if equivalent is None else equivalent.on(*operation.qubits)
        )
    reference_cost = clifford_t_counts(cirq, cirq.decompose(reference_operations))
    assert dynamic_cost == reference_cost
    print(
        f"FlagQuantum unitary: {local_workspace} workspace qubits, {dict(unitary_cost)}"
    )
    print(f"FlagQuantum dynamic: {workspace} workspace qubits, {dict(dynamic_cost)}")
    print(f"Qualtran: {len(ancillas)} workspace wires, {dict(reference_cost)}")
