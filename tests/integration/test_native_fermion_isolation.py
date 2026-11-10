"""A clean native workflow runs with chemistry and circuit SDK imports blocked."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_native_workflow_without_external_science_packages() -> None:
    script = """
        import importlib.abc
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "scipy", "h5py", "numpy"}
        class NoExternalScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalScience())
        import torch
        import flagquantum as fq
        from flagquantum.algorithms.chemistry_domain.problem import from_native_molecule, fermion_hamiltonian
        from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
        from flagquantum.algorithms.molecular import create_molecular_hamiltonian
        from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
        from flagquantum.algorithms.fermion.observables import to_observable
        from flagquantum.algorithms.fermion.low_rank import decompose_two_body
        from flagquantum.algorithms.fermion.evolution import low_rank_evolution
        from flagquantum.algorithms.fermion.preparation import prepare_slater, prepare_gaussian
        from flagquantum.algorithms.fermion.lowering import fermionic_fourier_transform, lift_stabilizer_state
        from flagquantum.algorithms.fermion.stabilizer import reduce_stabilizers
        from flagquantum.core.algebra.pauli import pauli
        from flagquantum.ecosystem.chemistry.pyscf import build_molecular_problem
        problem = from_native_molecule(create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))]))
        active = select_active_space(problem, active_orbitals=(0, 1)).problem
        reference = None
        for encoding in ENCODINGS:
            encoded = encode_fermion(fermion_hamiltonian(active), n_modes=4, encoding=encoding)
            circuit = fq.Circuit(4, dtype=torch.complex128)
            for wire, bit in enumerate(encoded.reference_bits(active.reference_occupation)):
                if bit:
                    circuit.x(wire)
            energy = float(torch.as_tensor(fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))]).expectation()).reshape(-1)[0])
            assert abs(energy - problem.reference_energy) < 1e-8
            if reference is not None:
                assert abs(energy - reference) < 1e-8
            reference = energy
        factors = decompose_two_body(active.two_body)
        state = prepare_slater(torch.eye(4, dtype=torch.complex128)[:, :2])
        time = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        state.compose(low_rank_evolution(active.one_body, factors, time, constant=active.energy_offset))
        gradient = torch.autograd.grad(state.state().reshape(-1)[12].real, time)[0]
        assert torch.isfinite(gradient)
        assert prepare_gaussian(torch.eye(2), torch.zeros(2, 2)).state().reshape(-1)[0].abs() > 0.999
        fourier_state = fq.Circuit(4, dtype=torch.complex128)
        fourier_state.x(1)
        original = fourier_state.state().clone()
        fourier_state.compose(fermionic_fourier_transform(4))
        torch.testing.assert_close(fourier_state.state().abs().square().sum(), torch.tensor(1.0, dtype=torch.float64))
        fourier_state.compose(fermionic_fourier_transform(4, inverse=True))
        torch.testing.assert_close(fourier_state.state(), original)
        symmetry = pauli(factors=((0, "y"), (1, "x")))
        reduced = reduce_stabilizers(symmetry, n_qubits=2, generators=[symmetry], eigenvalues=[-1])
        lifted = lift_stabilizer_state(reduced, torch.tensor([1, 0], dtype=torch.complex128))
        reduced_circuit = fq.Circuit(reduced.n_qubits, dtype=torch.complex128)
        energy = fq.run(reduced_circuit, outputs=[fq.expectation(to_observable(reduced.operator))]).expectation()
        assert abs(float(torch.as_tensor(energy).reshape(-1)[0]) + 1) < 1e-10
        assert lifted.shape == (4,)
        assert abs(float(lifted.abs().square().sum()) - 1) < 1e-10
        try:
            build_molecular_problem([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
        except ImportError as error:
            assert "chemistry-pyscf" in str(error)
        else:
            raise AssertionError("optional driver import was not blocked")
        from examples.algorithms.fermion_orbital_rotation import main as orbital_example
        orbital_example()
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
    """
    script = textwrap.dedent(script)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
