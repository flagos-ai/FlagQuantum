"""Measure one fermionic problem through three interchangeable native encodings.

Run: python -m examples.algorithms.fermion_energy
Expected: each encoding reports energy 1.3 without OpenFermion or Cirq.
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import number


def main() -> None:
    hamiltonian = 1.3 * number(0) - 0.4 * number(1)
    for encoding in ENCODINGS:
        encoded = encode_fermion(hamiltonian, n_modes=3, encoding=encoding)
        circuit = fq.Circuit(encoded.n_qubits)
        for wire, bit in enumerate(encoded.reference_bits((1, 0, 0))):
            if bit:
                circuit.x(wire)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        print(
            f"{encoding}: {float(torch.as_tensor(result.expectation()).reshape(-1)[0]):.6f}"
        )


if __name__ == "__main__":
    main()
