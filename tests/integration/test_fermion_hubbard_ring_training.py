"""Prove periodic UCCSD training against a separate two-particle lattice model."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_hubbard_ring_training import energy
from flagquantum.algorithms.chemistry import uccsd_ansatz, uccsd_excitations
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    fermi_hubbard_chain,
    fermi_hubbard_ring,
)
from flagquantum.algorithms.fermion.observables import to_observable

pytestmark = pytest.mark.integration


def _sector_energy(
    parameters: torch.Tensor, t: float, u: float, mu: float
) -> torch.Tensor:
    state = (
        uccsd_ansatz(6, 2, parameters, dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    # The distinguishable (up site, down site) basis differs from interleaved
    # canonical occupations by a minus sign when the down creation comes first.
    indices = [
        (1 << (5 - 2 * up)) | (1 << (4 - 2 * down))
        for up in range(3)
        for down in range(3)
    ]
    signs = torch.tensor(
        [1 if up <= down else -1 for up in range(3) for down in range(3)]
    )
    coefficients = state[indices] * signs
    hopping = -t * (
        torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)
    )
    matrix = torch.kron(hopping, torch.eye(3, dtype=torch.float64)) + torch.kron(
        torch.eye(3, dtype=torch.float64), hopping
    )
    matrix += torch.diag(
        torch.tensor(
            [u if i == j else 0 for i in range(3) for j in range(3)],
            dtype=torch.float64,
        )
    )
    matrix -= 2 * mu * torch.eye(9, dtype=torch.float64)
    return (coefficients.conj() @ matrix.to(torch.complex128) @ coefficients).real


@pytest.mark.parametrize("t,u,mu", [(1.0, 4.0, 0.0), (-0.7, 2.0, 0.3)])
def test_periodic_training_matches_independent_lattice_energy_and_gradients(
    t: float, u: float, mu: float
) -> None:
    count = uccsd_excitations(2, 6).parameter_count
    parameters = torch.linspace(
        -0.2, 0.3, count, dtype=torch.float32, requires_grad=True
    )
    hamiltonian = to_observable(
        encode_fermion(
            fermi_hubbard_ring(3, t=t, u=u, mu=mu),
            n_modes=6,
            encoding="jordan_wigner",
        ).operator
    )
    initial = energy(parameters, hamiltonian)
    torch.testing.assert_close(
        initial, _sector_energy(parameters, t, u, mu), atol=1e-12, rtol=0
    )
    gradient = torch.autograd.grad(initial, parameters)[0]
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    _sector_energy(parameters + shift, t, u, mu)
                    - _sector_energy(parameters - shift, t, u, mu)
                )
                / 2e-3
                for shift in 1e-3 * torch.eye(count, dtype=parameters.dtype)
            ]
        )
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.1
    torch.testing.assert_close(gradient.double(), differences, atol=3e-4, rtol=3e-4)
    before = parameters.detach().clone()
    optimizer = torch.optim.SGD([parameters], lr=0.02)
    optimizer.zero_grad()
    torch.autograd.backward(energy(parameters, hamiltonian))
    optimizer.step()
    assert not torch.equal(before, parameters.detach())
    assert float(energy(parameters, hamiltonian).detach()) < float(initial.detach())
    state = (
        uccsd_ansatz(6, 2, parameters, dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    sector = [
        i
        for i in range(64)
        if (i & 0b101010).bit_count() == 1 and (i & 0b010101).bit_count() == 1
    ]
    assert float(state[sector].abs().square().sum().detach()) == pytest.approx(
        1, abs=1e-12
    )
    outside = [i for i in range(64) if i not in sector]
    assert float(state[outside].abs().square().sum().detach()) < 1e-24
    chain = to_observable(
        encode_fermion(
            fermi_hubbard_chain(3, t=t, u=u, mu=mu), n_modes=6, encoding="jordan_wigner"
        ).operator
    )
    # Removing the closing bond changes the same consumer's value and gradient.
    ring_value = energy(parameters, hamiltonian)
    chain_value = energy(parameters, chain)
    assert float((ring_value - chain_value).abs().detach()) > 0.01
    ring_gradient = torch.autograd.grad(ring_value, parameters)[0]
    chain_gradient = torch.autograd.grad(chain_value, parameters)[0]
    assert float((ring_gradient - chain_gradient).norm()) > 0.1


def test_training_rejects_incorrect_parameter_count() -> None:
    with pytest.raises(ValueError, match="parameter"):
        uccsd_ansatz(6, 2, torch.zeros(3), dtype=torch.complex128, device="cpu")


def test_ring_training_runs_without_external_science_sdks() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py"}
        class NoExternalScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalScience())
        from examples.algorithms.fermion_hubbard_ring_training import main
        main()
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(
        line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line
    )
    assert float(values["Final energy"]) < float(values["Initial energy"]) - 1
    assert (
        float(values["Final energy"])
        >= float(values["Analytic N=2 ground energy"]) - 1e-6
    )
    assert float(values["Gradient residual"]) < 3e-4
    assert float(values["Parameter update norm"]) > 0.1
    assert float(values["Up particles"]) == pytest.approx(1, abs=1e-7)
    assert float(values["Down particles"]) == pytest.approx(1, abs=1e-7)
