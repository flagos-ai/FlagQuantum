"""Graph model assembly composes with native measurement and angle training."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_hubbard_graph import build_ansatz, energy
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    fermi_hubbard_chain,
    fermi_hubbard_graph,
    fermi_hubbard_ring,
)
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import number

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_branching_hop_sign_gradients_updates_and_populations(
    encoding: Encoding,
) -> None:
    operator = fermi_hubbard_graph(4, ((0, 1), (0, 2), (0, 3)), t=1.2, u=4, mu=0.2)
    observable = to_observable(
        encode_fermion(operator, n_modes=8, encoding=encoding).operator
    )
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value = energy(angle, observable, encoding=encoding)
    assert float(value.detach()) == pytest.approx(1.2 * math.sin(0.37) - 0.4, abs=1e-12)
    gradient = torch.autograd.grad(value, angle)[0]
    assert float(gradient) == pytest.approx(1.2 * math.cos(0.37), abs=1e-12)
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable, encoding=encoding)
            - energy(angle - 1e-5, observable, encoding=encoding)
        ) / 2e-5
    torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)
    optimizer = torch.optim.SGD([angle], lr=0.1)
    population_observables = [
        to_observable(
            encode_fermion(
                number(spin) + number(spin + 2) + number(spin + 4) + number(spin + 6),
                n_modes=8,
                encoding=encoding,
            ).operator
        )
        for spin in (0, 1)
    ]
    for _ in range(6):
        previous = energy(angle, observable, encoding=encoding)
        optimizer.zero_grad()
        torch.autograd.backward(previous)
        optimizer.step()
        torch.testing.assert_close(
            energy(angle, observable, encoding=encoding),
            1.2 * angle.sin() - 0.4,
            atol=1e-12,
            rtol=0,
        )
        assert float(energy(angle, observable, encoding=encoding).detach()) < float(
            previous.detach()
        )
        for population in population_observables:
            torch.testing.assert_close(
                energy(angle, population, encoding=encoding),
                angle.new_tensor(1),
                atol=1e-12,
                rtol=0,
            )
    missing = fermi_hubbard_graph(4, ((0, 1), (0, 2)), t=1.2, u=4, mu=0.2)
    missing_observable = to_observable(
        encode_fermion(missing, n_modes=8, encoding=encoding).operator
    )
    assert float(
        energy(angle, missing_observable, encoding=encoding).detach()
    ) == pytest.approx(-0.4)
    assert float(
        torch.autograd.grad(
            energy(angle, missing_observable, encoding=encoding), angle
        )[0]
    ) == pytest.approx(0)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("periodic", [False, True])
def test_path_and_cycle_replace_the_producer_at_the_training_consumer(
    periodic: bool,
    encoding: Encoding,
) -> None:
    edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3))
    if periodic:
        edges = (*edges, (3, 0))
    original = (fermi_hubbard_ring if periodic else fermi_hubbard_chain)(
        4, t=0.7, u=3, mu=0.1
    )
    replacement = fermi_hubbard_graph(4, edges, t=0.7, u=3, mu=0.1)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    observables = [
        to_observable(encode_fermion(op, n_modes=8, encoding=encoding).operator)
        for op in (original, replacement)
    ]
    values = [energy(angle, op, encoding=encoding) for op in observables]
    gradients = [torch.autograd.grad(value, angle)[0] for value in values]
    torch.testing.assert_close(values[0], values[1], atol=1e-12, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-12, rtol=0)
    assert abs(float(values[0].detach())) > 0.01
    if periodic:
        assert abs(float(gradients[0])) > 0.1


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_graph_cli_without_external_science_or_dense_eigensolvers(
    encoding: Encoding,
) -> None:
    script = """
        import importlib.abc
        import runpy
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoScience())
        import torch
        def refuse(*args, **kwargs):
            raise AssertionError("dense eigensolver used")
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            setattr(torch.linalg, name, refuse)
        sys.argv = ["fermion_hubbard_graph", "--encoding", sys.argv[1]]
        runpy.run_module("examples.algorithms.fermion_hubbard_graph", run_name="__main__")
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), encoding],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Encoding: {encoding}" in result.stdout
    assert "Alpha/beta populations: 1.000000, 1.000000" in result.stdout


@pytest.mark.parametrize(
    "encoding,first,second",
    [
        ("jordan_wigner", "00010010", "10010000"),
        ("parity", "00011100", "11100000"),
        ("bravyi_kitaev", "00010010", "11000000"),
    ],
)
@pytest.mark.parametrize("theta", [-1.2, 0.0, 0.37, 2.4])
def test_fixed_preparation_matches_independent_encoded_words(
    encoding: Encoding, first: str, second: str, theta: float
) -> None:
    angle = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    actual = build_ansatz(angle, encoding=encoding).state().reshape(-1)
    expected = torch.zeros(256, dtype=torch.complex128)
    expected[int(first, 2)] = math.cos(theta / 2)
    expected[int(second, 2)] = math.sin(theta / 2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    # Phase-sensitive amplitude derivatives protect the trainable preparation.
    first_gradient = torch.autograd.grad(
        actual[int(first, 2)].real, angle, retain_graph=True
    )[0]
    second_gradient = torch.autograd.grad(actual[int(second, 2)].real, angle)[0]
    assert float(first_gradient) == pytest.approx(-math.sin(theta / 2) / 2, abs=1e-12)
    assert float(second_gradient) == pytest.approx(math.cos(theta / 2) / 2, abs=1e-12)


@pytest.mark.parametrize("encoding", ["unknown"])
def test_preparation_refuses_unknown_basis(encoding: Encoding) -> None:
    with pytest.raises(ValueError, match="unsupported encoding"):
        build_ansatz(torch.tensor(0.37), encoding=encoding)


def test_preparation_refuses_nonscalar_angle() -> None:
    with pytest.raises(ValueError, match="scalar angle"):
        build_ansatz(torch.tensor([0.37]))
