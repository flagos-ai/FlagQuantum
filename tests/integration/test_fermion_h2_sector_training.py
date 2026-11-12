"""Molecular reduced training matches an independent occupation-space reference."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_h2_sector_training import (
    H2_FCIDUMP,
    build_ansatz,
    energy,
)
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    StabilizerReduction,
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.core.algebra.pauli import pauli
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import number
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.integration


def _native_problem() -> MolecularProblem:
    return from_native_molecule(
        create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    )


def _reduce(problem: MolecularProblem, *, up_parity: int = -1) -> StabilizerReduction:
    operator = encode_fermion(fermion_hamiltonian(problem), n_modes=4).operator
    candidates = [
        pauli(factors=((0, "z"), (2, "z"))),
        pauli(factors=((1, "z"), (3, "z"))),
    ]
    generators = discover_pauli_symmetries(operator, n_qubits=4, candidates=candidates)
    assert generators == tuple(candidates)
    return reduce_stabilizers(
        operator, n_qubits=4, generators=generators, eigenvalues=[up_parity, -1]
    )


def _physical_state(angle: torch.Tensor) -> torch.Tensor:
    # Explicit occupations, independent of encoding, reduction and Clifford maps.
    basis = torch.eye(16, dtype=torch.complex128)
    return angle.div(2).cos() * basis[12] + angle.div(2).sin() * basis[3]


@pytest.fixture(params=("fcidump", "native"))
def problem(request: pytest.FixtureRequest) -> MolecularProblem:
    if request.param == "fcidump":
        return read_fcidump(H2_FCIDUMP, max_orbitals=2)
    return _native_problem()


def test_training_matches_independent_molecular_sector(
    problem: MolecularProblem,
) -> None:
    reduced = _reduce(problem)
    assert reduced.n_qubits == 2
    observable = to_observable(reduced.operator)
    full = fock_matrix(fermion_hamiltonian(problem), 4)
    up = torch.tensor([((i >> 3) & 1) + ((i >> 1) & 1) for i in range(16)])
    down = torch.tensor([((i >> 2) & 1) + (i & 1) for i in range(16)])
    a, c, b = full[12, 12].real, full[3, 3].real, full[12, 3].real
    for value in (-0.8, 0.0, 0.2, 1.7):
        angle = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        state = _physical_state(angle)
        actual = energy(angle, observable)
        reference = torch.vdot(state, full @ state).real
        analytic = (a + c) / 2 + (a - c) * angle.cos() / 2 + b * angle.sin()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual, analytic, atol=1e-12, rtol=0)
        gradient = torch.autograd.grad(actual, angle)[0]
        torch.testing.assert_close(
            gradient, torch.autograd.grad(reference, angle)[0], atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            gradient, -(a - c) * angle.sin() / 2 + b * angle.cos(), atol=1e-12, rtol=0
        )
        with torch.no_grad():
            plus, minus = _physical_state(angle + 1e-5), _physical_state(angle - 1e-5)
            difference = (
                torch.vdot(plus, full @ plus).real
                - torch.vdot(minus, full @ minus).real
            ) / 2e-5
        torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)

    angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    reference_angle = angle.detach().clone().requires_grad_()
    optimizer = torch.optim.SGD([angle], lr=0.2)
    reference_optimizer = torch.optim.SGD([reference_angle], lr=0.2)
    populations = [
        to_observable(reduced.reduce_operator(encode_fermion(op, n_modes=4).operator))
        for op in (number(0) + number(2), number(1) + number(3))
    ]
    for _ in range(6):
        previous = energy(angle, observable)
        state = _physical_state(reference_angle)
        reference = torch.vdot(state, full @ state).real
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        torch.autograd.backward(previous)
        torch.autograd.backward(reference)
        assert angle.grad is not None and reference_angle.grad is not None
        torch.testing.assert_close(angle.grad, reference_angle.grad, atol=1e-12, rtol=0)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(angle, reference_angle, atol=1e-12, rtol=0)
        assert float(energy(angle, observable).detach()) < float(previous.detach())
        with torch.no_grad():
            lifted = lift_stabilizer_state(
                reduced, build_ansatz(angle).state().reshape(-1)
            )
            torch.testing.assert_close(
                lifted, _physical_state(reference_angle), atol=1e-12, rtol=0
            )
            torch.testing.assert_close(up * lifted, lifted)
            torch.testing.assert_close(down * lifted, lifted)
            outside = [i for i in range(16) if up[i] != 1 or down[i] != 1]
            assert float(lifted[outside].abs().square().sum()) < 1e-24
            for population in populations:
                torch.testing.assert_close(
                    energy(angle, population), angle.new_tensor(1)
                )

    minimum = (a + c) / 2 - torch.hypot((a - c) / 2, b)
    sector = [i for i in range(16) if i.bit_count() == 2]
    torch.testing.assert_close(
        minimum, torch.linalg.eigvalsh(full[sector][:, sector])[0], atol=1e-12, rtol=0
    )
    assert float(minimum) == pytest.approx(-1.1372701747, abs=1e-10)


def test_input_replacement_preserves_the_training_consumer() -> None:
    reductions = [
        _reduce(read_fcidump(H2_FCIDUMP, max_orbitals=2)),
        _reduce(_native_problem()),
    ]
    observables = [to_observable(reduction.operator) for reduction in reductions]
    angles = [
        torch.tensor(0.2, dtype=torch.float64, requires_grad=True) for _ in observables
    ]
    optimizers = [torch.optim.SGD([angle], lr=0.2) for angle in angles]
    for _ in range(6):
        values = [
            energy(angle, observable)
            for angle, observable in zip(angles, observables, strict=True)
        ]
        torch.testing.assert_close(values[0], values[1], atol=1e-12, rtol=0)
        for value, optimizer in zip(values, optimizers, strict=True):
            optimizer.zero_grad()
            torch.autograd.backward(value)
            optimizer.step()
        assert angles[0].grad is not None and angles[1].grad is not None
        torch.testing.assert_close(angles[0].grad, angles[1].grad, atol=1e-12, rtol=0)
        torch.testing.assert_close(angles[0], angles[1], atol=1e-12, rtol=0)


def test_opposite_spin_sector_is_not_a_two_electron_projection() -> None:
    reduced = _reduce(read_fcidump(H2_FCIDUMP, max_orbitals=2), up_parity=1)
    state = torch.ones(4, dtype=torch.complex128) / 2
    lifted = lift_stabilizer_state(reduced, state)
    population = torch.tensor([i.bit_count() for i in range(16)], dtype=torch.float64)
    weights = lifted.abs().square()
    # Mean N=2 alone is insufficient: this different sector has only N=1 and N=3.
    torch.testing.assert_close(weights @ population, population.new_tensor(2))
    assert float(weights[population == 2].sum()) == 0
    torch.testing.assert_close(
        weights @ (population - 2).square(), population.new_tensor(1)
    )


def test_invalid_preparation_and_nonpreserving_measurement_fail() -> None:
    reduced = _reduce(read_fcidump(H2_FCIDUMP, max_orbitals=2))
    with pytest.raises(ValueError, match="scalar angle"):
        build_ansatz(torch.tensor([0.2]))
    with pytest.raises(ValueError, match="does not preserve"):
        reduced.reduce_operator(pauli(factors=((0, "x"),)))
    state = build_ansatz(torch.tensor(0.2, requires_grad=True)).state().reshape(-1)
    with pytest.raises(
        ValueError, match="static preprocessing; autograd inputs are unsupported"
    ):
        lift_stabilizer_state(reduced, state)


def test_cli_runs_without_external_science_or_dense_eigenstate_preparation() -> None:
    script = """
        import importlib.abc
        import runpy
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoScience())
        from unittest.mock import patch
        with patch("torch.linalg.eigh", side_effect=AssertionError("dense eigensolver")), patch(
            "torch.linalg.eigvalsh", side_effect=AssertionError("dense eigensolver")
        ):
            runpy.run_module("examples.algorithms.fermion_h2_sector_training", run_name="__main__")
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
