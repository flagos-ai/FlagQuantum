"""Reduced training agrees with independent occupation-space pairing physics."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_parity_training import build_ansatz, energy
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.symmetry import (
    discover_z_parities,
    reduce_z_parities,
)
from flagquantum.observables import FermionOperator, FermionTerm
from flagquantum.observables.fermion import annihilate, create, number

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("coupling", [0.7, -0.4])
def test_encoding_replacement_preserves_sector_energy_gradient_and_update(
    encoding: Encoding, coupling: float
) -> None:
    identity = FermionOperator((FermionTerm(1),))
    model = (
        number(0)
        + number(1)
        - identity
        - coupling * (create(0) * create(1) + annihilate(1) * annihilate(0))
    )
    encoded = encode_fermion(model, n_modes=2, encoding=encoding)
    physical_parity = (identity - 2 * number(0)) * (identity - 2 * number(1))
    parity = encode_fermion(physical_parity, n_modes=2, encoding=encoding).operator
    word = tuple(q for q, _ in parity.terms[0].factors)
    discovered = discover_z_parities(encoded, candidates=[word])
    assert discovered == (word,)
    reduced = reduce_z_parities(encoded, generators=discovered, eigenvalues=[1])
    observable = to_observable(reduced.operator)
    assert reduced.n_qubits == 1
    # Independent occupation order |00>, |01>, |10>, |11>. The pair term
    # connects vacuum and double occupancy; odd states have zero energy.
    matrix = torch.tensor(
        [[-1, 0, 0, -coupling], [0, 0, 0, 0], [0, 0, 0, 0], [-coupling, 0, 0, 1]],
        dtype=torch.complex128,
    )
    indices = [
        sum(bit << (1 - q) for q, bit in enumerate(encoded.reference_bits(bits)))
        for bits in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]

    def physical_energy(angle: torch.Tensor) -> torch.Tensor:
        lifted = reduced.mapping.lift_state(build_ansatz(angle).state().reshape(-1))
        state = lifted[indices]
        torch.testing.assert_close(state[[1, 2]], torch.zeros(2, dtype=state.dtype))
        torch.testing.assert_close(
            state.abs().square().sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        return (state.conj() @ matrix @ state).real

    for value in (0.0, 1.7, -2.1):
        angle = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        actual = energy(angle, observable)
        reference = physical_energy(angle)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        gradient = torch.autograd.grad(actual, angle)[0]
        expected = torch.autograd.grad(reference, angle)[0]
        with torch.no_grad():
            difference = (
                physical_energy(angle + 1e-5) - physical_energy(angle - 1e-5)
            ) / 2e-5
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)

    angle = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    optimizer = torch.optim.SGD([angle], lr=0.05)
    for _ in range(5):
        previous = energy(angle, observable)
        optimizer.zero_grad()
        torch.autograd.backward(previous)
        optimizer.step()
        current = energy(angle, observable)
        assert float(current.detach()) < float(previous.detach())
        torch.testing.assert_close(current, physical_energy(angle), atol=1e-12, rtol=0)
    assert float((angle.detach() - before).abs()) > 0.01
    # Choosing the opposite sector changes the physical subspace, even though
    # the same one-qubit preparation and measurement consumers still work.
    odd = reduce_z_parities(encoded, generators=discovered, eigenvalues=[-1])
    odd_state = odd.mapping.lift_state(build_ansatz(angle).state().reshape(-1))[indices]
    torch.testing.assert_close(odd_state[[0, 3]], torch.zeros(2, dtype=odd_state.dtype))
    assert float((odd_state.conj() @ matrix @ odd_state).real.detach()) == 0


def test_preparation_rejects_vector_angles() -> None:
    with pytest.raises(ValueError, match="scalar angle"):
        build_ansatz(torch.tensor([0.2, 0.3]))


def test_training_without_external_quantum_or_chemistry_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        blocked = {"openfermion", "cirq", "pyscf", "scipy", "h5py"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoScience())
        from examples.algorithms.fermion_parity_training import main
        main()
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
