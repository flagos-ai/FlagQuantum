"""Run native measurement from a two-orbital FCIDUMP, without science SDKs."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "FCIDUMP"
        path.write_text(
            "&FCI NORB=2,NELEC=2,MS2=0,ORBSYM=1,1 /\n"
            "-1.0 1 1 0 0\n0.5 2 2 0 0\n0.4 1 1 1 1\n0.7 0 0 0 0\n",
            encoding="utf-8",
        )
        problem = read_fcidump(path)
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(
            fermion_hamiltonian(problem), n_modes=4, encoding=encoding
        )
        circuit = fq.Circuit(4, dtype=torch.complex128)
        for qubit, bit in enumerate(
            encoded.reference_bits(problem.reference_occupation)
        ):
            if bit:
                circuit.x(qubit)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        energy = float(torch.as_tensor(result.expectation()).reshape(-1)[0])
        assert abs(energy + 0.9) < 1e-10
        print(f"{encoding}: reference energy {energy:.10f} Ha")


if __name__ == "__main__":
    main()
