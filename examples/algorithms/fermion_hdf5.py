"""Read a MolecularData HDF5 file and execute native encoded measurements."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5


def main() -> None:
    """Create a small format fixture; no electronic-structure computation is run."""
    import h5py

    print(
        "premise: real closed-shell file integrals; the occupied reference is not a computed ground state.\n"
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "molecule.hdf5"
        with h5py.File(path, "w") as source:
            for name, value in {
                "n_orbitals": 2,
                "n_electrons": 2,
                "multiplicity": 1,
                "nuclear_repulsion": 0.5,
            }.items():
                source[name] = value
            source["one_body_integrals"] = [[-1.0, 0.0], [0.0, -0.2]]
            two = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
            two[0, 0, 0, 0] = 0.6
            source["two_body_integrals"] = two.numpy()
        problem = read_molecular_hdf5(path)
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(
            fermion_hamiltonian(problem), n_modes=4, encoding=encoding
        )
        circuit = fq.Circuit(4, dtype=torch.complex128)
        for wire, bit in enumerate(
            encoded.reference_bits(problem.reference_occupation)
        ):
            if bit:
                circuit.x(wire)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        energy = float(torch.as_tensor(result.expectation()).reshape(-1)[0])
        assert abs(energy + 0.9) < 1e-10
        print(f"{encoding}: reference energy {energy:.10f} Ha")


if __name__ == "__main__":
    main()
