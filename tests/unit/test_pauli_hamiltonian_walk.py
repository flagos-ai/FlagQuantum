"""Verify Pauli Hamiltonian preprocessing and the composed source walk."""

from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_requirements,
)
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    pauli_hamiltonian_walk,
    prepare_precision_from_eigenphase,
)
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


def test_nonreal_locked_norm_requires_explicit_bound() -> None:
    ham = Hamiltonian([HamiltonianTerm(1, "X", (0,)), HamiltonianTerm(-1, "Z", (0,))])
    with pytest.raises(ValueError, match="Source norm estimate is nonreal"):
        pauli_hamiltonian_walk(ham, 1)
    assert pauli_hamiltonian_walk(
        ham, 1, norm_bound=sqrt(2)
    ).selector.select_unitaries == ("X", "-Z")


@pytest.mark.ftqc_reference
def test_symbolic_precision_matches_locked_source() -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        get_prepare_precision_from_eigenphase_precision,
    )

    e, count, norm, bound = sympy.symbols("e L alpha beta", positive=True)
    assert (
        sympy.simplify(
            prepare_precision_from_eigenphase(e, count, norm, bound)
            - get_prepare_precision_from_eigenphase_precision(e, count, norm, bound)
        )
        == 0
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_signed_walk_from_terms_matches_locked_source(adjoint: bool) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        walk_operator_for_pauli_hamiltonian,
    )
    from qualtran.cirq_interop.testing import GateHelper

    q0, q1 = cirq.LineQubit(2), cirq.LineQubit(9)
    source_ham = cirq.PauliSum.from_pauli_strings(
        [3 * cirq.X(q1) * cirq.Z(q0), -3 * cirq.Y(q1) * cirq.Z(q0)]
    )
    ham = Hamiltonian(
        [
            HamiltonianTerm(0.75, {9: "X", 2: "Z"}),
            HamiltonianTerm(-3, {9: "Y", 2: "Z"}),
            HamiltonianTerm(2.25, {2: "Z", 9: "X"}),
        ]
    )
    norm = sqrt(18)
    original = walk_operator_for_pauli_hamiltonian(
        source_ham, 1, ham_norm_upper_bound=norm
    )
    source = original.adjoint() if adjoint else original
    plan = pauli_hamiltonian_walk(ham, 1, norm_bound=norm)
    assert plan.target_qubits == (2, 9)
    assert plan.selector.select_unitaries == ("ZX", "-ZY")
    assert plan.walk.alpha == original.sum_of_lcu_coefficients
    assert plan.table.alternates == tuple(original.prepare.alt)
    assert plan.table.keep == tuple(original.prepare.keep)
    assert plan.table.mu == original.prepare.mu == 2
    model = plan.resource_model(adjoint=adjoint)
    assert_source_model(model, source)
    stages = tuple(
        str(instance.bloq) for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    expected_stages = (
        "SelectBlockEncoding†" if adjoint else "SelectBlockEncoding",
        "ReflectionUsingPrepare",
    )
    assert stages == (expected_stages[::-1] if adjoint else expected_stages)

    helper = GateHelper(source)
    ports = {
        name: tuple(helper.operation.qubits.index(q) for q in wires.flat)
        for name, wires in helper.quregs.items()
    }
    width = len(helper.operation.qubits)
    work, slots = alias_preparation_requirements(plan.table)
    select_work = unary_iteration_workspace(len(ports["selection"]), None)
    reflect_work, reflect_slots = prepare_reflection_requirements(
        len(ports["selection"]), controlled=False
    )
    junk = plan.walk.block_encoding.signal_state.junk_width
    work_count = junk + max(select_work, work + 1, reflect_work)
    work_wires = tuple(range(width + junk, width + work_count))
    circuit = DynamicCircuit(width + work_count).append(
        plan.walk.make_on(
            selection={"selection": ports["selection"]},
            target=ports["target"],
            signal_junk=tuple(range(width, width + junk)),
            select_workspace=work_wires[:select_work],
            select_classical_bits=tuple(range(select_work)),
            prepare_workspace=work_wires[1 : 1 + work],
            prepare_classical_bits=tuple(range(slots)),
            reflection_workspace=work_wires[:reflect_work],
            reflection_classical_bits=tuple(range(reflect_slots)),
            adjoint=adjoint,
        )
    )

    actual = circuit.to_ir().instructions
    measurements = sum(gate.name == "measure" for gate in actual)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.logical.peak_logical_qubits == 14
    assert report.logical.gates.total_t_count() == 32
    assert report.emitted.t_count == 38 and measurements == 6
    initial = torch.zeros(1 << (width + work_count), dtype=torch.complex128)
    logical = torch.arange(1, 1 + (1 << width), dtype=torch.float64).to(
        torch.complex128
    )
    logical = (logical + 1j * logical.flip(0)) / sqrt(2 * float(logical.norm()) ** 2)
    initial[:: 1 << work_count] = logical
    # Retain the source reflection's allocated preparation junk so its
    # complete output can be compared, including any nonzero junk sectors.
    address = ports["selection"]
    target = ports["target"]
    logical_and_junk = width + junk
    q = cirq.LineQubit.range(logical_and_junk)
    prep_wires = tuple(q[i] for i in (*address, *range(width, logical_and_junk)))
    prep = cirq.MatrixGate(original.prepare.tensor_contract()).on(*prep_wires)
    select = cirq.MatrixGate(original.block_encoding.select.tensor_contract()).on(
        *(q[i] for i in (*address, *target))
    )
    reflect = cirq.Circuit(
        cirq.inverse(prep),
        cirq.Z(q[address[0]]),
        prep,
    )
    source_circuit = (
        reflect + cirq.Circuit(select) if adjoint else cirq.Circuit(select) + reflect
    )
    reference_input = np.zeros(1 << logical_and_junk, dtype=np.complex128)
    reference_input[:: 1 << junk] = logical.numpy()
    assert np.linalg.norm(reference_input) == pytest.approx(1)

    reference = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(source_circuit, qubit_order=q, initial_state=reference_input)
        .final_state_vector
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << len(work_wires)] = torch.from_numpy(reference) / sqrt(
        2**measurements
    )
    # The locked reflection can leave its signal junk dirty, even from zero.
    assert np.linalg.norm(reference.reshape(-1, 1 << junk)[:, 1:]) > 0.9
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, actual, history),
            expected,
            atol=3e-12,
            rtol=0,
        )
