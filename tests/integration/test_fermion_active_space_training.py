"""Frozen-core training agrees with an independent full-Fock-space projection."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_active_space_training import (
    energy,
    illustrative_problem,
)
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.integration


def test_projected_training_preserves_energies_gradients_and_updates() -> None:
    problem = illustrative_problem()
    selection = select_active_space(
        problem, active_orbitals=(2, 0), frozen_occupied=(1,)
    )
    active = selection.problem
    assert active.orbital_labels == (2, 0)
    assert active.reference_occupation == (1, 1, 0, 0)
    assert active.n_electrons == 2
    assert selection.discarded_virtual == (3,)
    observable = to_observable(
        encode_fermion(
            fermion_hamiltonian(active), n_modes=4, encoding="jordan_wigner"
        ).operator
    )
    full = fock_matrix(fermion_hamiltonian(problem), 8)
    # Embed creation products in the original mode order, including their
    # fermionic permutation signs. Freeze 1 doubly occupied and discard 3 empty.
    embedding = torch.zeros((256, 16), dtype=torch.complex128)
    modes = (4, 5, 0, 1)
    for column in range(16):
        occupied = [modes[q] for q in range(4) if column & (1 << (3 - q))]
        occupied += [2, 3]
        inversions = sum(
            a > b for i, a in enumerate(occupied) for b in occupied[i + 1 :]
        )
        row = sum(1 << (7 - q) for q in occupied)
        embedding[row, column] = (-1) ** inversions
    torch.testing.assert_close(
        embedding.mH @ embedding, torch.eye(16, dtype=torch.complex128)
    )
    reduced = fock_matrix(fermion_hamiltonian(active), 4)
    torch.testing.assert_close(embedding.mH @ full @ embedding, reduced)
    # The omitted virtual orbital really couples to retained states; this is
    # not an accidentally invariant full-space problem.
    discarded_rows = [i for i in range(256) if i & 3]
    assert float((full @ embedding)[discarded_rows].norm()) > 1e-3
    parameters = torch.tensor([0.1, -0.07, 0.2], requires_grad=True)
    projected_parameters = parameters.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    projected_optimizer = torch.optim.Adam([projected_parameters], lr=0.05)
    initial = float(energy(parameters, observable).detach())
    for step in range(25):
        optimizer.zero_grad()
        projected_optimizer.zero_grad()
        value = energy(parameters, observable)
        state = (
            uccsd_ansatz(
                4, 2, projected_parameters, dtype=torch.complex128, device="cpu"
            )
            .state()
            .reshape(-1)
        )
        lifted = embedding @ state
        reference = torch.vdot(lifted, full @ lifted).real
        torch.testing.assert_close(value, reference, atol=1e-11, rtol=0)
        torch.autograd.backward(value)
        torch.autograd.backward(reference)
        assert parameters.grad is not None
        assert projected_parameters.grad is not None
        torch.testing.assert_close(
            parameters.grad, projected_parameters.grad, atol=2e-7, rtol=0
        )
        if step == 0:
            assert torch.all(parameters.grad.abs() > 1e-3)
            with torch.no_grad():
                differences = torch.stack(
                    [
                        (
                            energy(parameters + shift, observable)
                            - energy(parameters - shift, observable)
                        )
                        / 2e-3
                        for shift in 1e-3 * torch.eye(3)
                    ]
                )
            torch.testing.assert_close(
                parameters.grad.double(), differences, atol=1e-6, rtol=1e-4
            )
        if step < 24:
            optimizer.step()
            projected_optimizer.step()
        torch.testing.assert_close(parameters, projected_parameters, atol=2e-6, rtol=0)
    assert float(value.detach()) < initial - 1e-2
    assert float(state.detach().abs().square().sum()) == pytest.approx(1, abs=1e-12)
    outside = [i for i in range(16) if i not in (3, 6, 9, 12)]
    assert float(state.detach()[outside].abs().square().sum()) < 1e-20
    outside_full = [
        i
        for i in range(256)
        if i.bit_count() != 4 or i & 0b00110000 != 0b00110000 or i & 3
    ]
    assert float(lifted.detach()[outside_full].abs().square().sum()) < 1e-20


def test_example_runs_without_external_science_sdks() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import runpy
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py"}
        class NoExternalScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalScience())
        runpy.run_module("examples.algorithms.fermion_active_space_training", run_name="__main__")
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
    """,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
