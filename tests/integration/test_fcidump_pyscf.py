"""Independent PySCF FCIDUMP producer with native-only consumers."""

import importlib.util
from pathlib import Path

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("pyscf") is None,
        reason="install flagquantum[chemistry-pyscf] for FCIDUMP producer tests",
    ),
]


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_pyscf_generated_integrals_and_reference_energy(
    tmp_path: Path, basis: str
) -> None:
    from pyscf import ao2mo, gto, scf
    from pyscf.tools import fcidump

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis=basis, verbose=0)
    mean_field = scf.RHF(molecule).run(conv_tol=1e-12)
    assert mean_field.converged
    path = tmp_path / "FCIDUMP"
    fcidump.from_scf(mean_field, str(path), tol=0.0, float_format="%.17g")
    problem = read_fcidump(path)
    orbitals = mean_field.mo_coeff
    expected_one = orbitals.T @ mean_field.get_hcore() @ orbitals
    expected_two = ao2mo.restore(
        1, ao2mo.kernel(molecule, orbitals), problem.n_orbitals
    )
    torch.testing.assert_close(problem.one_body, torch.as_tensor(expected_one))
    torch.testing.assert_close(problem.two_body, torch.as_tensor(expected_two))
    assert problem.energy_offset == pytest.approx(molecule.energy_nuc(), abs=1e-12)
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=problem.n_modes)
    matrix = pauli_matrix(encoded.operator, n_qubits=problem.n_modes)
    bits = encoded.reference_bits(problem.reference_occupation)
    index = int("".join(map(str, bits)), 2)
    assert float(matrix[index, index].real) == pytest.approx(mean_field.e_tot, abs=1e-9)
