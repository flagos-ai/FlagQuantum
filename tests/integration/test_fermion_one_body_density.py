"""Independent occupation action verifies native encoded density measurements."""

from __future__ import annotations

import itertools
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.algorithms.fermion_one_body_density import build_circuit
from flagquantum.algorithms.fermion.density import (
    one_body_density,
    one_body_density_observables,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import FermionOperator, annihilate, create

pytestmark = pytest.mark.integration


def _annihilated(state: torch.Tensor, mode: int, width: int) -> torch.Tensor:
    result = torch.zeros_like(state)
    mask = 1 << (width - 1 - mode)
    for source in range(state.numel()):
        if source & mask:
            preceding = source >> (width - mode)
            result[source ^ mask] = (-1) ** preceding.bit_count() * state[source]
    return result


def _oracle_values(state: torch.Tensor, width: int) -> tuple[torch.Tensor, ...]:
    lowered = [_annihilated(state, p, width) for p in range(width)]
    values = []
    for p in range(width):
        values.append(torch.vdot(lowered[p], lowered[p]).real)
        for q in range(p + 1, width):
            coherence = torch.vdot(lowered[p], lowered[q])
            values.extend((coherence.real, coherence.imag))
    return tuple(values)


def _encoded_state(state: torch.Tensor, width: int, encoding: Encoding) -> torch.Tensor:
    encoded = torch.zeros_like(state)
    for index in range(state.numel()):
        bits = [(index >> (width - 1 - p)) & 1 for p in range(width)]
        if encoding == "parity":
            bits = [sum(bits[: p + 1]) % 2 for p in range(width)]
        elif encoding == "bravyi_kitaev":
            # Fenwick node p+1 owns lowbit(p+1) consecutive occupations.
            bits = [
                sum(bits[(p + 1) - ((p + 1) & -(p + 1)) : p + 1]) % 2
                for p in range(width)
            ]
        target = sum(bit << (width - 1 - p) for p, bit in enumerate(bits))
        encoded[target] = state[index]
    return encoded


def _prepared(state: torch.Tensor, width: int, encoding: Encoding) -> fq.Circuit:
    # Test-only dense completion avoids the unrelated complex64 amplitude loader.
    assert width <= 4
    encoded = _encoded_state(state, width, encoding)
    basis = torch.eye(1 << width, dtype=torch.complex128)
    basis[:, 0] = encoded
    unitary = torch.linalg.qr(basis).Q
    unitary[:, 0] *= torch.vdot(unitary[:, 0], encoded)
    circuit = fq.Circuit(width, dtype=torch.complex128, device="cpu")
    circuit.any(*range(width), matrix=unitary)
    torch.testing.assert_close(circuit.state().reshape(-1), encoded, atol=1e-12, rtol=0)
    return circuit


def _measure(
    state: torch.Tensor, width: int, encoding: Encoding
) -> tuple[torch.Tensor, ...]:
    circuit = _prepared(state, width, encoding)
    observables = one_body_density_observables(n_modes=width, encoding=encoding)
    return fq.run(
        circuit, outputs=[fq.expectation(o) for o in observables]
    ).expectations


def _consume(
    values: tuple[torch.Tensor, ...], h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = one_body_density(values, n_modes=h.shape[0])
    return gamma, (h * gamma).sum()


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize(
    "kind", ["vacuum", "determinant", "complex_particle", "correlated"]
)
def test_occupation_oracle_replaced_by_encoded_expectations(
    kind: str, encoding: Encoding
) -> None:
    width = 4
    state = torch.zeros(16, dtype=torch.complex128)
    if kind == "vacuum":
        state[0] = 1
        particles = 0
    elif kind == "determinant":
        state[0b1011] = 1
        particles = 3
    elif kind == "complex_particle":
        state[0b1000], state[0b0001] = math.sqrt(0.3), 1j * math.sqrt(0.7)
        particles = 1
    else:
        state[0b1100], state[0b0011], state[0b1010] = (
            math.sqrt(0.4),
            1j * math.sqrt(0.35),
            math.sqrt(0.25),
        )
        particles = 2
    generator = torch.Generator().manual_seed(19)
    h = torch.randn(width, width, dtype=torch.complex128, generator=generator)
    h = h + h.mH
    reference, energy = _consume(_oracle_values(state, width), h)
    measured, reconstructed = _consume(_measure(state, width, encoding), h)
    torch.testing.assert_close(measured, reference, atol=1e-12, rtol=0)
    torch.testing.assert_close(reconstructed, energy, atol=1e-12, rtol=0)
    torch.testing.assert_close(measured, measured.mH, atol=1e-12, rtol=0)
    assert measured.trace().real.item() == pytest.approx(particles, abs=1e-12)
    spectrum = torch.linalg.eigvalsh(measured)
    assert spectrum.min() >= -1e-12
    assert spectrum.max() <= 1 + 1e-12
    # Complete one-body energy uses an independently applied fermionic action.
    operator = FermionOperator()
    for p, q in itertools.product(range(width), repeat=2):
        operator += complex(h[p, q]) * create(p) * annihilate(q)
    circuit = _prepared(state, width, encoding)
    observable = to_observable(
        encode_fermion(operator, n_modes=width, encoding=encoding).operator
    )
    direct = (
        fq.run(circuit, outputs=[fq.expectation(observable)]).expectation().reshape(())
    )
    torch.testing.assert_close(reconstructed.real, direct, atol=1e-12, rtol=0)
    assert abs(reconstructed.imag.item()) < 1e-12
    if kind == "correlated":
        assert torch.linalg.matrix_norm(measured @ measured - measured) > 0.1


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_complex_orbital_rotation_by_independent_two_particle_minors(
    encoding: Encoding,
) -> None:
    width = 4
    state = torch.zeros(16, dtype=torch.complex128)
    state[0b1100], state[0b0011] = math.sqrt(0.4), 1j * math.sqrt(0.6)
    generator = torch.Generator().manual_seed(27)
    unitary = torch.linalg.qr(
        torch.randn(width, width, dtype=torch.complex128, generator=generator)
    ).Q
    occupations = list(itertools.combinations(range(width), 2))
    indices = [sum(1 << (width - 1 - p) for p in modes) for modes in occupations]
    rotated = torch.zeros_like(state)
    for target, rows in zip(indices, occupations, strict=True):
        for source, cols in zip(indices, occupations, strict=True):
            rotated[target] += (
                torch.linalg.det(unitary[list(rows)][:, list(cols)]) * state[source]
            )
    original = one_body_density(_measure(state, width, encoding), n_modes=width)
    actual = one_body_density(_measure(rotated, width, encoding), n_modes=width)
    assert actual.trace().real.item() == pytest.approx(2, abs=1e-12)
    torch.testing.assert_close(
        original,
        torch.diag(torch.tensor([0.4, 0.4, 0.6, 0.6], dtype=torch.complex128)),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        actual, unitary.conj() @ original @ unitary.T, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_complex_coherence_energy_gradients_and_optimizer(encoding: Encoding) -> None:
    angle = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    observables = one_body_density_observables(n_modes=2, encoding=encoding)
    h = torch.tensor([[0.7, 0.3j], [-0.3j, -0.2]], dtype=torch.complex128)

    def loss(theta: torch.Tensor) -> torch.Tensor:
        values = fq.run(
            build_circuit(theta, encoding=encoding),
            outputs=[fq.expectation(o) for o in observables],
        ).expectations
        gamma, value = _consume(values, h)
        expected = torch.stack(
            (
                torch.stack(
                    (
                        torch.cos(theta / 2).square().to(torch.complex128),
                        0.5j * theta.sin(),
                    )
                ),
                torch.stack(
                    (
                        -0.5j * theta.sin(),
                        torch.sin(theta / 2).square().to(torch.complex128),
                    )
                ),
            )
        )
        torch.testing.assert_close(gamma, expected, atol=1e-12, rtol=0)
        return value.real

    value = loss(angle)
    gradient = torch.autograd.grad(value, angle)[0]
    expected_gradient = -0.45 * angle.sin() - 0.3 * angle.cos()
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=0)
    difference = (loss(angle.detach() + 1e-5) - loss(angle.detach() - 1e-5)) / 2e-5
    torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)
    optimizer = torch.optim.SGD([angle], lr=0.1)
    optimizer.zero_grad()
    loss(angle).backward()
    optimizer.step()
    assert loss(angle).item() < value.item()


def test_measurement_width_mismatch_fails_during_planning() -> None:
    observables = one_body_density_observables(n_modes=2)
    with pytest.raises(ValueError, match="qubit|register|range|width"):
        fq.plan(fq.Circuit(1), outputs=[fq.expectation(o) for o in observables])


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_cli_without_openfermion_cirq_or_external_science(encoding: Encoding) -> None:
    script = """
        import importlib.abc
        import runpy
        import sys
        blocked = {"openfermion", "cirq", "pyscf", "openfermionpyscf", "scipy", "h5py"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoScience())
        import torch
        def refuse(*args, **kwargs):
            raise AssertionError("dense inspection used in normal execution")
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            setattr(torch.linalg, name, refuse)
        sys.argv = ["fermion_one_body_density", "--encoding", sys.argv[1]]
        runpy.run_module("examples.algorithms.fermion_one_body_density", run_name="__main__")
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), encoding],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Particle number: 1.0000000000" in result.stdout
    assert "One-body energy: 0.4520082847" in result.stdout
    assert "Angle gradient: -0.5016897975" in result.stdout
