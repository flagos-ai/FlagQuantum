"""Measure N=2 on a two-site Hubbard chain or three-site periodic ring.

Run: python -m examples.algorithms.fermion_hubbard [--periodic]
"""

import argparse

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    fermi_hubbard_chain,
    fermi_hubbard_ring,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.observables.fermion import FermionOperator, number


def main(*, periodic: bool = False) -> None:
    profile = "three-site periodic ring" if periodic else "two-site open chain"
    print(
        f"premise: fixed {profile} N=2 CPU sector; dense diagonalization and state "
        "preparation have exponential inspection cost, not a many-body scalability claim.\n"
    )
    sites = 3 if periodic else 2
    width = 2 * sites
    hamiltonian = (
        fermi_hubbard_ring(sites, t=1, u=4)
        if periodic
        else fermi_hubbard_chain(sites, t=1, u=4)
    )
    particles = sum((number(mode) for mode in range(width)), FermionOperator())
    jw = encode_fermion(hamiltonian, n_modes=width, encoding="jordan_wigner")
    matrix = pauli_matrix(jw.operator, n_qubits=width)
    sector = [i for i in range(2**width) if i.bit_count() == 2]
    energies, vectors = torch.linalg.eigh(matrix[sector][:, sector])
    reference = torch.zeros(2**width, dtype=torch.complex128)
    reference[sector] = vectors[:, 0]
    analytic = (2 - 68**0.5) / 2 if periodic else (4 - 32**0.5) / 2
    print(f"Analytic N=2 energy: {analytic:.10f}")
    print(f"Dense N=2 energy: {float(energies[0]):.10f}")
    for encoding in ENCODINGS:
        encoded = encode_fermion(hamiltonian, n_modes=width, encoding=encoding)
        population = encode_fermion(particles, n_modes=width, encoding=encoding)
        state = torch.zeros_like(reference)
        for index, amplitude in enumerate(reference):
            occupation = tuple((index >> (width - 1 - q)) & 1 for q in range(width))
            bits = encoded.reference_bits(occupation)
            target = sum(bit << (width - 1 - q) for q, bit in enumerate(bits))
            state[target] = amplitude
        circuit = fq.Circuit(width, dtype=torch.complex128)
        append_arbitrary_state(circuit, state, qubits=tuple(range(width)))
        result = fq.run(
            circuit,
            outputs=[
                fq.expectation(to_observable(encoded.operator)),
                fq.expectation(to_observable(population.operator)),
            ],
        )
        energy, count = (
            float(torch.as_tensor(value).reshape(-1)[0])
            for value in result.expectations
        )
        print(f"{encoding} energy: {energy:.10f}; particles: {count:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--periodic", action="store_true", help="inspect a three-site ring"
    )
    main(periodic=parser.parse_args().periodic)
