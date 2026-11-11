"""File input composes with existing UCCSD and differentiable native execution."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from examples.algorithms.fermion_fcidump_training import DEFAULT_FCIDUMP, energy, main
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

pytestmark = pytest.mark.integration


def test_file_and_native_integral_producers_share_the_same_training_consumer() -> None:
    problem = read_fcidump(DEFAULT_FCIDUMP)
    native = create_molecular_hamiltonian(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    )
    torch.testing.assert_close(problem.one_body, native.one_body, atol=1e-12, rtol=0)
    torch.testing.assert_close(problem.two_body, native.two_body, atol=1e-12, rtol=0)
    assert problem.energy_offset == pytest.approx(native.nuclear_repulsion, abs=1e-12)
    operator = to_observable(
        encode_fermion(
            fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner"
        ).operator
    )
    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    file_energy = energy(parameters, operator)
    native_energy = energy(parameters, native.hamiltonian)
    torch.testing.assert_close(file_energy, native_energy, atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(file_energy, parameters)[0]
    native_gradient = torch.autograd.grad(native_energy, parameters)[0]
    torch.testing.assert_close(gradient, native_gradient, atol=1e-7, rtol=0)
    assert torch.all(gradient.abs() > 1e-3)
    differences = []
    with torch.no_grad():
        for shift in 1e-3 * torch.eye(3, dtype=torch.float32):
            differences.append(
                (
                    energy(parameters + shift, native.hamiltonian)
                    - energy(parameters - shift, native.hamiltonian)
                )
                / 2e-3
            )
    torch.testing.assert_close(
        gradient.to(torch.float64), torch.stack(differences), atol=1e-6, rtol=1e-4
    )
    state = (
        uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
        .state()
        .detach()
        .reshape(-1)
    )
    # One alpha and one beta particle: (up_0, down_0, up_1, down_1).
    outside = [index for index in range(16) if index not in (3, 6, 9, 12)]
    assert float(state[outside].abs().square().sum()) < 1e-20
    assert float(state.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_imported_constant_shifts_energy_once_without_changing_gradients(
    tmp_path: Path,
) -> None:
    original = read_fcidump(DEFAULT_FCIDUMP)
    lines = DEFAULT_FCIDUMP.read_text(encoding="utf-8").splitlines()
    lines[-1] = f"{original.energy_offset + 0.4:.17e} 0 0 0 0"
    path = tmp_path / "shifted.fcidump"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shifted = read_fcidump(path)
    parameters = torch.tensor([0.1, -0.07, 0.2], requires_grad=True)
    values = []
    gradients = []
    for problem in (original, shifted):
        observable = to_observable(
            encode_fermion(
                fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner"
            ).operator
        )
        value = energy(parameters, observable)
        values.append(value.detach())
        gradients.append(torch.autograd.grad(value, parameters)[0])
    assert float(values[1] - values[0]) == pytest.approx(0.4, abs=1e-12)
    torch.testing.assert_close(gradients[1], gradients[0], atol=1e-7, rtol=0)


@pytest.mark.parametrize("orbitals,electrons", [(1, 2), (2, 0), (2, 4), (3, 2)])
def test_example_refuses_inputs_outside_its_bounded_profile(
    tmp_path: Path, orbitals: int, electrons: int
) -> None:
    path = tmp_path / "FCIDUMP"
    path.write_text(
        f"&FCI NORB={orbitals},NELEC={electrons},MS2=0 /\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="two orbitals and two electrons|budget"):
        main(path)


def test_missing_input_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(tmp_path / "missing.fcidump")


def test_complete_training_without_external_science_sdks() -> None:
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
        runpy.run_module("examples.algorithms.fermion_fcidump_training", run_name="__main__")
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
