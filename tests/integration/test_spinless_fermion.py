"""Spinless closing bonds reach native measurement, gradients and updates."""

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    spinless_fermion_chain,
    spinless_fermion_ring,
)
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable

pytestmark = pytest.mark.integration


def _energy(theta: torch.Tensor, observable: Observable) -> torch.Tensor:
    # Mode 1 is occupied, reversing hopping from 3 to 0. The states are
    # |0101> and |1100>, each with exactly two particles.
    circuit = fq.Circuit(4, dtype=torch.complex128).x(1).x(3)
    circuit.ry(0, theta=theta).cx(0, 3)
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


@pytest.mark.parametrize("periodic", [False, True])
def test_interaction_hopping_gradients_and_optimizer(periodic: bool) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    t, v, mu = 1.2, 2.3, 0.2
    observable = to_observable(
        encode_fermion(builder(4, t=t, v=v, mu=mu), n_modes=4).operator
    )
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value = _energy(theta, observable)
    expected = v * (1 - math.cos(0.37)) / 2 - 2 * mu
    expected_gradient = v * math.sin(0.37) / 2
    if periodic:
        expected += t * math.sin(0.37)
        expected_gradient += t * math.cos(0.37)
    assert float(value.detach()) == pytest.approx(expected, abs=1e-12)
    gradient = torch.autograd.grad(value, theta)[0]
    assert float(gradient) == pytest.approx(expected_gradient, abs=1e-12)
    with torch.no_grad():
        difference = (
            _energy(theta + 1e-5, observable) - _energy(theta - 1e-5, observable)
        ) / 2e-5
    torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)
    optimizer = torch.optim.SGD([theta], lr=0.03)
    optimizer.zero_grad()
    torch.autograd.backward(_energy(theta, observable))
    optimizer.step()
    assert float(_energy(theta, observable).detach()) < expected


def test_both_spinless_workflows_without_external_science() -> None:
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
        from examples.algorithms.fermion_spinless import main
        main()
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
