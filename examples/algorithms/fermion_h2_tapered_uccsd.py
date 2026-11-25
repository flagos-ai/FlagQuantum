"""Train the existing H2 UCCSD factors in an explicit two-qubit spin sector.

Run: python -m examples.algorithms.fermion_h2_tapered_uccsd --encoding parity
"""

import argparse
from itertools import combinations

import torch

import flagquantum as fq
from examples.algorithms.fermion_fcidump_training import DEFAULT_FCIDUMP, train_adam
from flagquantum.algorithms.chemistry import excitation_operator, uccsd_excitations
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.core import Hamiltonian, pauli_term
from flagquantum.algorithms.fermion.clifford import pauli_words_commute
from flagquantum.algorithms.fermion.encoding import (
    ENCODINGS,
    Encoding,
    encode_fermion,
)
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.symmetry import ReducedOperator, reduce_z_parities
from flagquantum.algorithms.trotter import trotter_circuit
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, FermionTerm, number


def _h2_spin_parities(encoding: Encoding) -> tuple[tuple[int, ...], ...]:
    identity = FermionOperator((FermionTerm(1),))
    parities = []
    for left, right in ((0, 2), (1, 3)):
        parity = (identity - 2 * number(left)) * (identity - 2 * number(right))
        terms = encode_fermion(parity, n_modes=4, encoding=encoding).operator.terms
        if (
            len(terms) != 1
            or terms[0].coefficient != 1
            or not terms[0].factors
            or any(axis != "z" for _, axis in terms[0].factors)
        ):
            raise ValueError("expected a positive encoded Z word for each spin parity")
        parities.append(tuple(q for q, _ in terms[0].factors))
    return tuple(parities)


def prepare_h2(
    encoding: Encoding = "jordan_wigner",
) -> tuple[ReducedOperator, tuple[ReducedOperator, ...], tuple[int, ...]]:
    """Reduce the Hamiltonian and ordered UCCSD generators in one encoded sector."""
    problem = read_fcidump(DEFAULT_FCIDUMP, max_orbitals=2)
    parities = _h2_spin_parities(encoding)
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4, encoding=encoding)
    # Odd parity in each two-mode spin register fixes one electron of each spin.
    hamiltonian = reduce_z_parities(encoded, generators=parities, eigenvalues=(-1, -1))
    generators = []
    for factor in uccsd_excitations(2, 4).factors:
        midpoint = len(factor) // 2
        # Match exp(i * theta * H / 2) and the original virtual-index order.
        generator = 1j * excitation_operator(factor[:midpoint], factor[midpoint:])
        generators.append(
            reduce_z_parities(
                encode_fermion(generator, n_modes=4, encoding=encoding),
                generators=parities,
                eigenvalues=(-1, -1),
            )
        )
    return hamiltonian, tuple(generators), encoded.reference_bits((1, 1, 0, 0))


def build_ansatz(
    parameters: torch.Tensor,
    hamiltonian: ReducedOperator,
    generators: tuple[ReducedOperator, ...],
    reference: tuple[int, ...],
) -> fq.Circuit:
    """Prepare the reduced Hartree–Fock reference and exactly evolve each factor."""
    if (
        parameters.shape != (3,)
        or parameters.device.type != "cpu"
        or parameters.dtype != torch.float32
        or not bool(torch.isfinite(parameters).all())
    ):
        raise ValueError("expected three finite float32 CPU angles")
    mapping = hamiltonian.mapping
    if (
        hamiltonian.n_qubits != 2
        or len(generators) != 3
        or any(generator.mapping != mapping for generator in generators)
    ):
        raise ValueError(
            "H2 UCCSD requires three generators in the same two-qubit sector"
        )
    if len(reference) != mapping.n_qubits:
        raise ValueError(
            "reference occupation width does not match the selected sector"
        )
    bits = tuple(reference[q] for q in mapping.free_qubits)
    if mapping.full_bits(bits) != reference:
        raise ValueError("reference occupation does not belong to the selected sector")
    circuit = fq.Circuit(2, dtype=torch.complex128, device="cpu")
    for q, bit in enumerate(bits):
        if bit:
            circuit.x(q)
    for angle, generator in zip(parameters, generators, strict=True):
        terms = generator.operator.terms
        if any(not term.factors or term.coefficient.imag != 0 for term in terms) or any(
            not pauli_words_commute(left, right)
            for left, right in combinations(terms, 2)
        ):
            raise ValueError(
                "reduced factor requires commuting, nonidentity Hermitian Pauli terms"
            )
        if not terms:
            continue
        weighted = Hamiltonian(
            [
                pauli_term(-0.5 * term.coefficient.real * angle, dict(term.factors))
                for term in terms
            ]
        )
        # Within a factor the words commute; one product step is exact.
        circuit.compose(
            trotter_circuit(weighted, 1.0, order=1, n_qubits=2, dtype=torch.complex128)
        )
    return circuit


def energy(
    parameters: torch.Tensor,
    hamiltonian: ReducedOperator,
    generators: tuple[ReducedOperator, ...],
    reference: tuple[int, ...],
    observable: Observable,
) -> torch.Tensor:
    """Evaluate the reduced circuit using the native expectation consumer."""
    result: torch.Tensor = fq.run(
        build_ansatz(parameters, hamiltonian, generators, reference),
        outputs=[fq.expectation(observable)],
    ).expectation()
    return result.reshape(())


def main(encoding: Encoding = "jordan_wigner") -> None:
    hamiltonian, generators, reference = prepare_h2(encoding)
    observable = to_observable(hamiltonian.operator)

    def evaluate(angles: torch.Tensor) -> torch.Tensor:
        return energy(angles, hamiltonian, generators, reference, observable)

    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    before = parameters.detach().clone()
    initial = evaluate(parameters)
    gradient = torch.autograd.grad(initial, parameters)[0]
    train_adam(parameters, evaluate)
    with torch.no_grad():
        final = evaluate(parameters)
    print(
        "premise: exact CPU two-qubit statevector, fixed H2 fixture and "
        "explicit odd spin parities; no automatic sector choice or scalability claim.\n"
    )
    print(f"Encoding: {encoding}")
    print(f"Register: 4 -> {hamiltonian.n_qubits} qubits")
    print(f"Initial energy: {initial.detach().item():.10f} Ha")
    print(f"Gradient norm: {gradient.norm().item():.8f}")
    print(f"Parameter update norm: {(parameters.detach() - before).norm().item():.8f}")
    print(f"Final energy: {final.item():.10f} Ha")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    main(parser.parse_args().encoding)
