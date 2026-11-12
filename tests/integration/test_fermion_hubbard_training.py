"""Exercise sector preparation, runtime gradients and an actual optimizer step."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_hubbard_training import build_ansatz, energy
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("angles", [[0.4, -0.3, 0.2], [1.7, 2.1, -0.6], [0, 0, 0]])
def test_preparation_has_exactly_one_particle_of_each_spin(angles: list[float]) -> None:
    state = build_ansatz(torch.tensor(angles, dtype=torch.float64)).state().reshape(-1)
    # Big-endian JW modes are (up_0, down_0, up_1, down_1).
    sector = [3, 6, 9, 12]
    outside = [index for index in range(16) if index not in sector]
    assert float(state[outside].abs().square().sum()) < 1e-25
    assert float(state[sector].abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_runtime_gradient_matches_dense_difference_and_optimizer_updates() -> None:
    encoded = encode_fermion(
        fermi_hubbard_chain(2, t=1, u=4), n_modes=4, encoding="jordan_wigner"
    )
    observable = to_observable(encoded.operator)
    matrix = pauli_matrix(encoded.operator, n_qubits=4)
    parameters = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64, requires_grad=True)
    initial_parameters = parameters.detach().clone()
    initial = energy(parameters, observable)
    initial.backward()
    assert parameters.grad is not None
    assert torch.all(parameters.grad.abs() > 1e-3)
    differences = []
    with torch.no_grad():
        for shift in 1e-5 * torch.eye(3, dtype=torch.float64):
            plus = build_ansatz(parameters + shift).state().reshape(-1)
            minus = build_ansatz(parameters - shift).state().reshape(-1)
            differences.append(
                (
                    (plus.conj() @ matrix @ plus).real
                    - (minus.conj() @ matrix @ minus).real
                )
                / 2e-5
            )
    torch.testing.assert_close(
        parameters.grad, torch.stack(differences), atol=1e-8, rtol=1e-8
    )
    optimizer = torch.optim.Adam([parameters], lr=0.1)
    optimizer.step()
    assert torch.all((parameters.detach() - initial_parameters).abs() > 0.09)
    assert float(energy(parameters, observable).detach()) < float(initial.detach())


def test_training_without_external_quantum_or_chemistry_sdks() -> None:
    # Torch optimizer initialization can import NumPy through Dynamo. Keep the
    # stronger algebra-only no-NumPy test in test_native_fermion_isolation.py.
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
        from examples.algorithms.fermion_hubbard_training import main
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
