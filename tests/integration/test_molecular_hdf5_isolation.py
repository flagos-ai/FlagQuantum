"""Optional HDF5 imports stop at the file boundary."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5
from tests.unit.fermion.test_fcidump import _dump

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _training_trace(
    problem: MolecularProblem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observable = to_observable(
        encode_fermion(
            fermion_hamiltonian(problem),
            n_modes=problem.n_modes,
            encoding="jordan_wigner",
        ).operator
    )
    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    initial_parameters = parameters.detach().clone()
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    history = []
    initial_gradient = None
    for _ in range(120):
        optimizer.zero_grad()
        circuit = uccsd_ansatz(
            problem.n_modes,
            problem.n_electrons,
            parameters,
            dtype=torch.complex128,
            device="cpu",
        )
        value = (
            fq.run(circuit, outputs=[fq.expectation(observable)])
            .expectation()
            .reshape(())
        )
        history.append(value.detach())
        value.backward()
        assert parameters.grad is not None
        if initial_gradient is None:
            initial_gradient = parameters.grad.detach().clone()
        optimizer.step()
    circuit = uccsd_ansatz(
        problem.n_modes,
        problem.n_electrons,
        parameters,
        dtype=torch.complex128,
        device="cpu",
    )
    final = (
        fq.run(circuit, outputs=[fq.expectation(observable)]).expectation().reshape(())
    )
    history.append(final.detach())
    assert initial_gradient is not None and torch.all(initial_gradient.abs() > 1e-3)
    assert float((parameters.detach() - initial_parameters).norm()) > 0.1
    assert float(history[-1]) < float(history[0]) - 1e-3
    assert float(final.detach()) == pytest.approx(-1.1372701747, abs=1e-6)
    state = circuit.state().detach().reshape(-1)
    outside = [index for index in range(16) if index not in (3, 6, 9, 12)]
    assert float(state[outside].abs().square().sum()) < 1e-20
    assert float(state.abs().square().sum()) == pytest.approx(1, abs=1e-12)
    return torch.stack(history), parameters.detach(), initial_gradient


def _check_training_input_replacement(directory: Path) -> None:
    h5py = pytest.importorskip("h5py")
    native = from_native_molecule(
        create_molecular_hamiltonian(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        )
    )
    fcidump = directory / "FCIDUMP"
    fcidump.write_text(_dump(native), encoding="utf-8")
    hdf5 = directory / "molecule.hdf5"
    with h5py.File(hdf5, "w") as source:
        source["n_orbitals"] = native.n_orbitals
        source["n_electrons"] = native.n_electrons
        source["multiplicity"] = 1
        source["nuclear_repulsion"] = native.energy_offset
        source["one_body_integrals"] = native.one_body.numpy()
        # The file stores (ps|qr), while native tensors store (pq|rs).
        source["two_body_integrals"] = native.two_body.permute(0, 2, 3, 1).numpy()
    file_problems = (
        read_fcidump(fcidump, max_orbitals=2),
        read_molecular_hdf5(hdf5, max_orbitals=2),
    )
    baseline = _training_trace(native)
    for problem in file_problems:
        assert problem.reference_energy is None
        assert problem.reference_occupation == native.reference_occupation
        assert problem.energy_offset == pytest.approx(native.energy_offset, abs=1e-12)
        torch.testing.assert_close(
            problem.one_body, native.one_body, atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            problem.two_body, native.two_body, atol=1e-12, rtol=0
        )
        actual = _training_trace(problem)
        for value, expected in zip(actual, baseline, strict=True):
            torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-7)


def test_reader_import_and_native_workflow_without_hdf5() -> None:
    script = """
        import importlib.abc
        import sys
        blocked={"numpy","h5py","scipy","pyscf","openfermion","openfermionpyscf","cirq"}
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0,Blocked())
        from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5
        from examples.algorithms.fermion_chemistry import main
        try:
            read_molecular_hdf5("not-opened")
        except ImportError as error:
            assert "chemistry-hdf5" in str(error)
        else:
            raise AssertionError("missing optional dependency accepted")
        main()
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="requires chemistry-hdf5"
)
def test_file_measurement_without_circuit_or_chemistry_sdks() -> None:
    script = """
        import importlib.abc
        import sys
        blocked={"scipy","pyscf","openfermion","openfermionpyscf","cirq"}
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0,Blocked())
        from examples.algorithms.fermion_hdf5 import main
        main()
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("reference energy -0.9000000000 Ha") == 3


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="requires chemistry-hdf5"
)
def test_hdf5_input_replacement_preserves_training_without_sdks(tmp_path: Path) -> None:
    script = """
        import importlib.abc
        import sys
        from pathlib import Path
        blocked={"scipy","pyscf","openfermion","openfermionpyscf","cirq"}
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0,Blocked())
        from tests.integration.test_molecular_hdf5_isolation import _check_training_input_replacement
        _check_training_input_replacement(Path(sys.argv[1]))
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
