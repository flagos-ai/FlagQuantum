"""Spinless training preserves its sector and the closing fermionic bond."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_spinless_training import build_ansatz, energy
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    spinless_fermion_chain,
    spinless_fermion_ring,
)
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable

pytestmark = pytest.mark.integration


def _observable(periodic: bool, t: float, v: float, mu: float) -> Observable:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    return to_observable(
        encode_fermion(
            builder(3, t=t, v=v, mu=mu), n_modes=3, encoding="jordan_wigner"
        ).operator
    )


def _sector_energy(
    parameters: torch.Tensor, periodic: bool, t: float, v: float, mu: float
) -> torch.Tensor:
    # Independent occupation basis |011>, |101>, |110>. Adjacent hops have
    # no spectator; the closing 0--2 hop crosses one occupied mode and is +t.
    # Only |101> acquires the closing density interaction V.
    matrix = torch.tensor(
        [
            [v, -t, t if periodic else 0],
            [-t, v if periodic else 0, -t],
            [t if periodic else 0, -t, v],
        ],
        dtype=torch.complex128,
    ) - 2 * mu * torch.eye(3, dtype=torch.complex128)
    coefficients = build_ansatz(parameters).state().reshape(-1)[[3, 5, 6]]
    return (coefficients.conj() @ matrix @ coefficients).real


def _assert_two_particles(parameters: torch.Tensor) -> None:
    state = build_ansatz(parameters).state().reshape(-1)
    assert float(state[[0, 1, 2, 4, 7]].abs().square().sum().detach()) < 1e-25
    assert float(state[[3, 5, 6]].abs().square().sum().detach()) == pytest.approx(
        1, abs=1e-12
    )


@pytest.mark.parametrize("angles", [[0, 0], [0.7, -0.4], [2.1, -1.3]])
def test_preparation_stays_in_the_two_particle_sector(angles: list[float]) -> None:
    _assert_two_particles(torch.tensor(angles, dtype=torch.float64))


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("t,v,mu", [(1.0, 2.0, 0.0), (-0.7, -1.3, 0.2), (0.2, 2, -0.3)])
def test_same_training_consumer_matches_independent_sector_and_updates(
    periodic: bool, t: float, v: float, mu: float
) -> None:
    observable = _observable(periodic, t, v, mu)
    parameters = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    initial = energy(parameters, observable)
    torch.testing.assert_close(
        initial, _sector_energy(parameters, periodic, t, v, mu), atol=1e-12, rtol=0
    )
    gradient = torch.autograd.grad(initial, parameters)[0]
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    _sector_energy(parameters + shift, periodic, t, v, mu)
                    - _sector_energy(parameters - shift, periodic, t, v, mu)
                )
                / 2e-5
                for shift in 1e-5 * torch.eye(2, dtype=parameters.dtype)
            ]
        )
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, differences, atol=1e-9, rtol=0)
    before = parameters.detach().clone()
    optimizer = torch.optim.SGD([parameters], lr=0.03)
    for _ in range(4):
        previous = energy(parameters, observable)
        optimizer.zero_grad()
        previous.backward()
        optimizer.step()
        current = energy(parameters, observable)
        assert float(current.detach()) < float(previous.detach())
        torch.testing.assert_close(
            current, _sector_energy(parameters, periodic, t, v, mu), atol=1e-12, rtol=0
        )
        _assert_two_particles(parameters)
    assert float((parameters.detach() - before).norm()) > 1e-3


def test_constant_ring_sector_has_zero_gradient_and_no_sgd_update() -> None:
    # With t=0 every two-particle state on the three-site ring has one V bond.
    observable = _observable(True, 0.0, 2.0, -0.3)
    parameters = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    before = parameters.detach().clone()
    value = energy(parameters, observable)
    assert float(value.detach()) == pytest.approx(2.6, abs=1e-12)
    value.backward()
    assert parameters.grad is not None
    torch.testing.assert_close(
        parameters.grad, torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.optim.SGD([parameters], lr=0.03).step()
    torch.testing.assert_close(parameters.detach(), before, atol=1e-12, rtol=0)
    _assert_two_particles(parameters)


@pytest.mark.parametrize("t,v", [(1.2, 0.0), (0.0, 2.3)])
def test_closing_hop_and_density_independently_change_value_and_gradient(
    t: float, v: float
) -> None:
    parameters = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    difference = energy(parameters, _observable(True, t, v, 0.0)) - energy(
        parameters, _observable(False, t, v, 0.0)
    )
    amplitudes = build_ansatz(parameters).state().reshape(-1)
    expected = (
        v * amplitudes[5].abs().square()
        + 2 * t * (amplitudes[3].conj() * amplitudes[6]).real
    )
    torch.testing.assert_close(difference, expected, atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(difference, parameters)[0]
    assert float(difference.abs().detach()) > 0.01
    assert float(gradient.norm()) > 0.01


@pytest.mark.parametrize("shape", [(1,), (3,), (1, 2)])
def test_wrong_parameter_count_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="exactly two angles"):
        build_ansatz(torch.zeros(shape, dtype=torch.float64))


def test_both_training_workflows_without_external_science_sdks() -> None:
    # PyTorch optimizer initialization can load NumPy through Dynamo. Native
    # algebra and the nontraining example retain their stronger no-NumPy test.
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
        from examples.algorithms.fermion_spinless_training import main
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
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
