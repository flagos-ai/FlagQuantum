"""Periodic boundary hopping composes with native execution and autograd."""

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_ring
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable

pytestmark = pytest.mark.integration


def _energy(theta: torch.Tensor, observable: Observable) -> torch.Tensor:
    # One down-spin spectator reverses the boundary hopping matrix element.
    circuit = fq.Circuit(6, dtype=torch.complex128).x(1).x(4)
    circuit.ry(0, theta=theta).cx(0, 4)
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


def test_boundary_sign_reaches_expectations_gradients_and_an_optimizer_step() -> None:
    t, u, mu = 1.2, 4.0, 0.2
    encoded = encode_fermion(
        fermi_hubbard_ring(3, t=t, u=u, mu=mu), n_modes=6, encoding="jordan_wigner"
    )
    observable = to_observable(encoded.operator)
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value = _energy(theta, observable)
    expected = u * (1 - math.cos(0.37)) / 2 + t * math.sin(0.37) - 2 * mu
    assert float(value.detach()) == pytest.approx(expected, abs=1e-12)
    gradient = torch.autograd.grad(value, theta)[0]
    assert float(gradient) == pytest.approx(
        u * math.sin(0.37) / 2 + t * math.cos(0.37), abs=1e-12
    )
    with torch.no_grad():
        difference = (
            _energy(theta + 1e-5, observable) - _energy(theta - 1e-5, observable)
        ) / 2e-5
    torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)
    optimizer = torch.optim.SGD([theta], lr=0.03)
    optimizer.zero_grad()
    torch.autograd.backward(_energy(theta, observable))
    optimizer.step()
    assert float(_energy(theta, observable).detach()) < expected - 0.05


def test_periodic_example_without_external_science_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py", "numpy"}
        class NoExternalScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalScience())
        from examples.algorithms.fermion_hubbard import main
        main(periodic=True)
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
