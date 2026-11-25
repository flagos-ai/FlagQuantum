"""A direct independent RHF producer checks file order and measured energy."""

import importlib.util
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("pyscf") is None,
        reason="requires chemistry-pyscf producer",
    ),
]


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_independent_rhf_file_reference(tmp_path: Path, basis: str) -> None:
    import h5py
    from pyscf import ao2mo, gto, scf

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis=basis, verbose=0)
    solve = scf.RHF(molecule)
    solve.conv_tol = 1e-11
    expected = float(solve.kernel())
    assert solve.converged
    coeff = solve.mo_coeff
    n = coeff.shape[1]
    one = coeff.T @ solve.get_hcore() @ coeff
    two = ao2mo.kernel(molecule, coeff, compact=False).reshape(n, n, n, n)
    path = tmp_path / "independent.hdf5"
    with h5py.File(path, "w") as source:
        for name, value in {
            "n_orbitals": n,
            "n_electrons": molecule.nelectron,
            "multiplicity": 1,
            "nuclear_repulsion": molecule.energy_nuc(),
            "hf_energy": expected,
        }.items():
            source[name] = value
        source["one_body_integrals"] = one
        source["two_body_integrals"] = two.transpose(0, 2, 3, 1)
    problem = read_molecular_hdf5(path)
    assert problem.reference_energy == expected
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(
            fermion_hamiltonian(problem), n_modes=problem.n_modes, encoding=encoding
        )
        circuit = fq.Circuit(problem.n_modes, dtype=torch.complex128)
        for wire, bit in enumerate(
            encoded.reference_bits(problem.reference_occupation)
        ):
            if bit:
                circuit.x(wire)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        measured = float(torch.as_tensor(result.expectation()).reshape(-1)[0])
        assert measured == pytest.approx(expected, abs=1e-9)
