"""Verify Pauli Hamiltonian preprocessing and the composed source walk."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from itertools import product
from math import sqrt
from typing import TYPE_CHECKING

import pytest
import torch

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_requirements,
)
from flagquantum.algorithms.primitives.ising_walk import ising_walk
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    pauli_hamiltonian_walk,
    prepare_precision_from_eigenphase,
    source_pauli_hamiltonian_norm_bound,
    symbolic_pauli_hamiltonian_walk,
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

if TYPE_CHECKING:
    import cirq

pytestmark = pytest.mark.unit


def test_nonreal_locked_norm_requires_explicit_bound() -> None:
    ham = Hamiltonian([HamiltonianTerm(1, "X", (0,)), HamiltonianTerm(-1, "Z", (0,))])
    with pytest.raises(ValueError, match="Source norm estimate is nonreal"):
        source_pauli_hamiltonian_norm_bound(ham)
    with pytest.raises(ValueError, match="Source norm estimate is nonreal"):
        pauli_hamiltonian_walk(ham, 1)
    assert pauli_hamiltonian_walk(
        ham, 1, norm_bound=sqrt(2)
    ).selector.select_unitaries == ("X", "-Z")


@pytest.mark.ftqc_reference
def test_pauli_norm_and_symbolic_bound_match_locked_source() -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        upper_bound_norm_for_pauli_hamiltonian,
        walk_operator_for_pauli_hamiltonian,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    wire = cirq.LineQubit(4)
    source_ham = cirq.PauliSum.from_pauli_strings([4 * cirq.X(wire), -cirq.Z(wire)])
    hamiltonian = Hamiltonian([HamiltonianTerm(4, "X", 4), HamiltonianTerm(-1, "Z", 4)])
    signed_bound = upper_bound_norm_for_pauli_hamiltonian(source_ham)
    # The signed source formula can underestimate the true norm of mixed signs.
    assert source_pauli_hamiltonian_norm_bound(hamiltonian) == signed_bound == 1
    bound = sympy.Symbol("norm", positive=True)
    source = walk_operator_for_pauli_hamiltonian(
        source_ham, 0.25, ham_norm_upper_bound=bound
    )
    plan = symbolic_pauli_hamiltonian_walk(hamiltonian, 0.25, norm_bound=bound)
    assert plan.target_qubits == (4,)
    assert sympy.simplify(plan.prepare.probability_bits - source.prepare.mu) == 0
    gates = plan.gate_cost()
    for key, expected in get_cost_value(source, QECGatesCost()).asdict().items():
        assert sympy.simplify(getattr(gates, key) - expected) == 0


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


def _native_terms(
    words: Sequence[cirq.PauliString], labels: dict[cirq.Qid, int]
) -> Hamiltonian:
    """Bind external wires to test labels, retaining raw terms before merging."""
    return Hamiltonian(
        HamiltonianTerm(word.coefficient, {labels[q]: str(p) for q, p in word.items()})
        for word in words
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", ["duplicates", "cancelled_wire", "four_rows"])
def test_external_pauli_sum_has_the_same_native_walk(case: str) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        walk_operator_for_pauli_hamiltonian,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    named = cirq.NamedQubit("z")
    grid = cirq.GridQubit(2, 3)
    x, y, z = cirq.X(named), cirq.Y(grid), cirq.Z(named)
    if case == "duplicates":
        words = [3 * x * y, -2 * z, cirq.PauliString(coefficient=0.25), -x * y]
        select_words = ("YX", "-IZ", "II")
        targets = (3, 8)
    elif case == "cancelled_wire":
        words = [cirq.PauliString(coefficient=-0.5), 2 * y, 3 * x, z, -2 * y, -x]
        select_words = ("-I", "X", "Z")
        targets = (8,)
    else:
        words = [-2 * z * y, 2 * x, 0.25 * y, cirq.PauliString(coefficient=0.5)]
        select_words = ("-YZ", "IX", "YI", "II")
        targets = (3, 8)
    source_ham = cirq.PauliSum.from_pauli_strings(words)
    hamiltonian = _native_terms(words, {grid: 3, named: 8})
    norm = float(max(abs(np.linalg.eigvalsh(source_ham.matrix()))))
    source = walk_operator_for_pauli_hamiltonian(
        source_ham, 0.2, ham_norm_upper_bound=norm
    )
    native = pauli_hamiltonian_walk(hamiltonian, 0.2, norm_bound=norm)
    assert tuple({grid: 3, named: 8}[q] for q in sorted(source_ham.qubits)) == targets
    assert native.target_qubits == targets
    assert (
        native.selector.select_unitaries
        == select_words
        == tuple(
            ("-" if word.coefficient.real < 0 else "")
            + "".join("IXYZ"[p] for p in word.pauli_mask)
            for word in source.block_encoding.select.select_unitaries
        )
    )
    assert native.table.mu == source.prepare.mu == 4
    assert native.table.keep == tuple(source.prepare.keep)
    assert native.table.alternates == tuple(source.prepare.alt)
    model = native.resource_model()
    assert_source_model(model, source)

    e = sympy.Symbol("e", positive=True)
    symbolic = symbolic_pauli_hamiltonian_walk(hamiltonian, e, norm_bound=norm)
    source_symbolic = walk_operator_for_pauli_hamiltonian(
        source_ham, e, ham_norm_upper_bound=norm
    )
    gates = symbolic.gate_cost()
    for name, count in get_cost_value(source_symbolic, QECGatesCost()).asdict().items():
        assert sympy.simplify(getattr(gates, name) - count) == 0
    assert symbolic.peak_logical_qubits().subs(e, 0.2) == (
        25 if case == "cancelled_wire" else 26
    )


@pytest.mark.ftqc_reference
def test_external_input_boundaries_remain_explicit() -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        walk_operator_for_pauli_hamiltonian,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value

    q = cirq.NamedQubit("z")
    singleton = [cirq.X(q)]
    source = walk_operator_for_pauli_hamiltonian(
        cirq.PauliSum.from_pauli_strings(singleton), 1, ham_norm_upper_bound=0
    )
    assert source.block_encoding.select.selection_bitsize == 0
    with pytest.raises(RuntimeError, match="Selection register must not be empty"):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="at least two nonzero products"):
        pauli_hamiltonian_walk(_native_terms(singleton, {q: 3}), 1, norm_bound=0)

    nonhermitian = [(3 + 1j) * cirq.X(q), 3 * cirq.Z(q)]
    source = walk_operator_for_pauli_hamiltonian(
        cirq.PauliSum.from_pauli_strings(nonhermitian), 1, ham_norm_upper_bound=1
    )
    assert source.sum_of_lcu_coefficients == 6
    assert source.block_encoding.select.select_unitaries[0].coefficient == 3 + 1j
    with pytest.raises(ValueError, match="concrete finite real scalars"):
        pauli_hamiltonian_walk(_native_terms(nonhermitian, {q: 3}), 1, norm_bound=1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("fixture", ["ising", "signed", "mixed_labels"])
def test_signed_walk_from_terms_matches_locked_source(
    adjoint: bool, fixture: str
) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        get_walk_operator_for_1d_ising_model,
        walk_operator_for_pauli_hamiltonian,
    )
    from qualtran.cirq_interop.testing import GateHelper
    from quimb.tensor import contract_strategy

    if fixture == "ising":
        original, _ = get_walk_operator_for_1d_ising_model(
            1, 1.0, j_zz_strength=3, gamma_x_strength=-3
        )
        plan, _ = ising_walk(1, 1.0, j_zz=3, gamma_x=-3)
        assert plan.target_qubits == (0,)
        assert plan.selector.select_unitaries == ("Z", "-X")
    elif fixture == "signed":
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
        plan = pauli_hamiltonian_walk(ham, 1, norm_bound=norm)
        assert plan.target_qubits == (2, 9)
        assert plan.selector.select_unitaries == ("ZX", "-ZY")
    else:
        named, grid = cirq.NamedQubit("z"), cirq.GridQubit(2, 3)
        x, y, z = cirq.X(named), cirq.Y(named), cirq.Z(named)
        other = cirq.Y(grid)
        words = [
            2 * x * other,
            -z * other,
            cirq.PauliString(coefficient=0.25),
            y * other,
            -x * other,
        ]
        source_ham = cirq.PauliSum.from_pauli_strings(words)
        norm = sqrt(3) + 0.25
        original = walk_operator_for_pauli_hamiltonian(
            source_ham, 1, ham_norm_upper_bound=norm
        )
        plan = pauli_hamiltonian_walk(
            _native_terms(words, {grid: 3, named: 8}), 1, norm_bound=norm
        )
        assert plan.target_qubits == (3, 8)
        assert plan.selector.select_unitaries == ("YX", "-YZ", "II", "YY")
    source = original.adjoint() if adjoint else original
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
    step = partial(
        plan.walk.make_on,
        selection={"selection": ports["selection"]},
        target=ports["target"],
        signal_junk=tuple(range(width, width + junk)),
        select_workspace=work_wires[:select_work],
        select_classical_bits=tuple(range(select_work)),
        prepare_workspace=work_wires[1 : 1 + work],
        prepare_classical_bits=tuple(range(slots)),
        reflection_workspace=work_wires[:reflect_work],
        reflection_classical_bits=tuple(range(reflect_slots)),
    )
    circuit = DynamicCircuit(width + work_count).append(step(adjoint=adjoint))

    actual = circuit.to_ir().instructions
    measurements = sum(gate.name == "measure" for gate in actual)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    if fixture != "mixed_labels":
        assert report.logical.peak_logical_qubits == (13 if fixture == "ising" else 14)
        assert report.logical.gates.total_t_count() == 32
        assert report.emitted.t_count == 38 and measurements == 6
    else:
        assert report.logical.peak_logical_qubits == 16
        assert report.logical.gates.total_t_count() == 68
        assert report.emitted.instruction_qubits == 17
        assert report.emitted.t_count == 80 and measurements == 13
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
    with contract_strategy("greedy"):
        prep_matrix = original.prepare.tensor_contract()
        select_matrix = original.block_encoding.select.tensor_contract()
    prep = cirq.MatrixGate(prep_matrix).on(*prep_wires)
    unprep = cirq.MatrixGate(prep_matrix.conj().T).on(*prep_wires)
    select = cirq.MatrixGate(select_matrix).on(*(q[i] for i in (*address, *target)))
    zero_mark = np.diag([1] + [-1] * ((1 << len(address)) - 1))
    mark = cirq.MatrixGate(zero_mark).on(*(q[i] for i in address))
    reflect = cirq.Circuit(unprep, mark, prep)
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
    assert np.linalg.norm(reference.reshape(-1, 1 << junk)[:, 1:]) > (
        0.9 if fixture == "signed" else 0.1
    )
    histories = (
        tuple(product((0, 1), repeat=measurements))
        if fixture != "mixed_labels"
        else (
            (0,) * measurements,
            (1,) * measurements,
            tuple(i % 2 for i in range(measurements)),
            tuple(1 - i % 2 for i in range(measurements)),
        )
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(initial, actual, history),
            expected,
            atol=3e-12,
            rtol=0,
        )

    paired = circuit.append(step(adjoint=not adjoint)).to_ir().instructions
    assert paired[: len(actual)] == actual
    inverse = paired[len(actual) :]
    inverse_measurements = sum(gate.name == "measure" for gate in inverse)
    assert inverse_measurements == measurements
    # The checked histories share one unnormalized intermediate vector.
    # Check the inverse histories on that vector to cover their Cartesian product.
    recovered = initial / sqrt(2 ** (measurements + inverse_measurements))
    for history in histories:
        torch.testing.assert_close(
            forced_branch(expected, inverse, history), recovered, atol=3e-12, rtol=0
        )
