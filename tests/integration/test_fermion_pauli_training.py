"""Reduced training agrees with independent mixed-axis physical-sector states."""

import itertools
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_pauli_training import build_ansatz, energy
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.core.algebra.pauli import pauli

pytestmark = pytest.mark.integration


def _physical_matrices(
    coupling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Independent Kronecker products in physical qubit order, without native
    # Pauli conversion, reduction maps or Clifford lowering.
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    y_matrix = torch.kron(torch.kron(y, identity), identity)
    xz_matrix = torch.kron(torch.kron(identity, x), z)
    full = (
        torch.kron(torch.kron(identity, identity), z)
        + coupling * torch.kron(torch.kron(identity, z), x)
        + 0.2 * y_matrix
        + 0.3 * xz_matrix
    )
    return full, y_matrix, xz_matrix


def _physical_state(angle: torch.Tensor, sectors: tuple[int, int]) -> torch.Tensor:
    # Y0=s0 and X1=s1*(-1)**q2 define the selected joint sector explicitly.
    y_state = torch.tensor([1, sectors[0] * 1j], dtype=torch.complex128) / math.sqrt(2)
    basis = torch.eye(2, dtype=torch.complex128)
    columns = [
        torch.kron(
            torch.kron(
                y_state,
                torch.tensor([1, sectors[1] * (-1) ** bit], dtype=torch.complex128)
                / math.sqrt(2),
            ),
            basis[bit],
        )
        for bit in (0, 1)
    ]

    return torch.cos(angle / 2) * columns[0] + torch.sin(angle / 2) * columns[1]


@pytest.mark.parametrize("sectors", tuple(itertools.product((-1, 1), repeat=2)))
@pytest.mark.parametrize("coupling", (0.4, -0.6))
def test_joint_sector_training_matches_independent_physical_states(
    sectors: tuple[int, int], coupling: float
) -> None:
    y0 = pauli(factors=((0, "y"),))
    x1z2 = pauli(factors=((1, "x"), (2, "z")))
    operator = (
        pauli(factors=((2, "z"),))
        + coupling * pauli(factors=((1, "z"), (2, "x")))
        + 0.2 * y0
        + 0.3 * x1z2
    )
    discovered = discover_pauli_symmetries(operator, n_qubits=3)
    reduced = reduce_stabilizers(
        operator, n_qubits=3, generators=discovered, eigenvalues=sectors
    )
    signed = reduce_stabilizers(
        operator,
        n_qubits=3,
        generators=[-y0, x1z2],
        eigenvalues=[-sectors[0], sectors[1]],
    )
    assert reduced == signed
    assert reduced.n_qubits == 1
    observable = to_observable(reduced.operator)

    full, y_matrix, xz_matrix = _physical_matrices(coupling)

    def inspect(angle: torch.Tensor) -> torch.Tensor:
        state = _physical_state(angle, sectors)
        torch.testing.assert_close(y_matrix @ state, sectors[0] * state)
        torch.testing.assert_close(xz_matrix @ state, sectors[1] * state)
        torch.testing.assert_close(torch.vdot(state, state).real, angle.new_tensor(1))
        with torch.no_grad():
            lifted = lift_stabilizer_state(
                reduced, build_ansatz(angle).state().reshape(-1)
            )
            torch.testing.assert_close(lifted, state.detach(), atol=1e-12, rtol=0)
        return torch.vdot(state, full @ state).real

    for value in (0.0, 1.7, -2.1):
        angle = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        actual = energy(angle, observable)
        reference = inspect(angle)
        analytic = (
            0.2 * sectors[0] + 0.3 * sectors[1] + angle.cos() + coupling * angle.sin()
        )
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual, analytic, atol=1e-12, rtol=0)
        gradient = torch.autograd.grad(actual, angle)[0]
        expected = torch.autograd.grad(reference, angle)[0]
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            gradient, -angle.sin() + coupling * angle.cos(), atol=1e-12, rtol=0
        )
        with torch.no_grad():
            difference = (inspect(angle + 1e-5) - inspect(angle - 1e-5)) / 2e-5
        torch.testing.assert_close(gradient, difference, atol=1e-9, rtol=0)

    angle = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    optimizer = torch.optim.SGD([angle], lr=0.1)
    for _ in range(5):
        previous = energy(angle, observable)
        optimizer.zero_grad()
        torch.autograd.backward(previous)
        optimizer.step()
        current = energy(angle, observable)
        assert float(current.detach()) < float(previous.detach())
        torch.testing.assert_close(current, inspect(angle), atol=1e-12, rtol=0)
    assert float((angle.detach() - before).abs()) > 0.1
    opposite = reduce_stabilizers(
        operator,
        n_qubits=3,
        generators=discovered,
        eigenvalues=[-sectors[0], sectors[1]],
    )
    with torch.no_grad():
        other = lift_stabilizer_state(opposite, build_ansatz(angle).state().reshape(-1))
        torch.testing.assert_close(y_matrix @ other, -sectors[0] * other)
        torch.testing.assert_close(
            torch.vdot(_physical_state(angle, sectors), other).abs(),
            angle.new_tensor(0),
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            energy(angle, to_observable(opposite.operator)) - energy(angle, observable),
            angle.new_tensor(-0.4 * sectors[0]),
            atol=1e-12,
            rtol=0,
        )


def test_invalid_preparation_and_sector_inputs_fail_before_training() -> None:
    with pytest.raises(ValueError, match="scalar angle"):
        build_ansatz(torch.tensor([0.2, 0.3]))
    generator = pauli(factors=((0, "y"),))
    with pytest.raises(ValueError, match="one eigenvalue is required"):
        reduce_stabilizers(
            generator, n_qubits=2, generators=[generator], eigenvalues=[]
        )
    reduced = reduce_stabilizers(
        generator, n_qubits=2, generators=[generator], eigenvalues=[1]
    )
    state = build_ansatz(torch.tensor(0.2, requires_grad=True)).state().reshape(-1)
    with pytest.raises(
        ValueError, match="static preprocessing; autograd inputs are unsupported"
    ):
        lift_stabilizer_state(reduced, state)


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
        from unittest.mock import patch
        from examples.algorithms.fermion_pauli_training import main
        with patch("torch.linalg.eigh", side_effect=AssertionError("dense eigensolver")), patch(
            "torch.linalg.eigvalsh", side_effect=AssertionError("dense eigensolver")
        ):
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
