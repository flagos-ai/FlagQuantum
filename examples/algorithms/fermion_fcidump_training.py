"""Train the existing UCCSD circuit from a bounded two-orbital FCIDUMP.

Run: python -m examples.algorithms.fermion_fcidump_training [FCIDUMP]
"""

import argparse
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, number

DEFAULT_FCIDUMP = Path(__file__).with_name("data") / "h2_sto3g.fcidump"


def energy(parameters: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure existing UCCSD on the full four-mode JW register."""
    circuit = uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
    return (
        fq.run(circuit, outputs=[fq.expectation(hamiltonian)]).expectation().reshape(())
    )


def main(path: Path = DEFAULT_FCIDUMP) -> None:
    print(
        "premise: two-orbital closed-shell CPU training in full JW order; "
        "UCCSD angles are float32 and the dense N=2 reference has exponential "
        "inspection cost, not a general convergence or scalability claim.\n"
    )
    problem = read_fcidump(path, max_orbitals=2)
    if problem.n_orbitals != 2 or problem.n_electrons != 2:
        raise ValueError(
            "this training example requires two orbitals and two electrons"
        )
    encoded = encode_fermion(
        fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner"
    )
    observable = to_observable(encoded.operator)
    matrix = pauli_matrix(encoded.operator, n_qubits=4)
    sector = [index for index in range(16) if index.bit_count() == 2]
    reference = torch.linalg.eigvalsh(matrix[sector][:, sector])[0]
    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    initial_parameters = parameters.detach().clone()
    initial = energy(parameters, observable)
    gradient = torch.autograd.grad(initial, parameters)[0]
    # The existing UCCSD boundary normalizes angles to float32, including when
    # supplied as float64. Resolve differences above its quantization scale.
    epsilon = 1e-3
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    energy(parameters + shift, observable)
                    - energy(parameters - shift, observable)
                )
                / (2 * epsilon)
                for shift in epsilon * torch.eye(3, dtype=parameters.dtype)
            ]
        )
        determinant = energy(torch.zeros_like(parameters), observable)
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    for _ in range(160):
        optimizer.zero_grad()
        energy(parameters, observable).backward()
        optimizer.step()
    particles = sum((number(mode) for mode in range(4)), FermionOperator())
    population = to_observable(
        encode_fermion(particles, n_modes=4, encoding="jordan_wigner").operator
    )
    with torch.no_grad():
        final = energy(parameters, observable)
        circuit = uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
        count = fq.run(circuit, outputs=[fq.expectation(population)]).expectation()
    print(f"Reference determinant: {float(determinant):.10f} Ha")
    print(f"Initial energy: {float(initial.detach()):.10f} Ha")
    print(f"Gradient norm: {float(gradient.norm()):.8f}")
    print(f"Gradient residual: {float((gradient - differences).abs().max()):.3e}")
    print(
        f"Parameter update norm: {float((parameters.detach() - initial_parameters).norm()):.8f}"
    )
    print(f"Final energy: {float(final):.10f} Ha")
    print(f"Dense N=2 energy: {float(reference):.10f} Ha")
    print(f"Particles: {float(count.reshape(())):.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump", nargs="?", type=Path, default=DEFAULT_FCIDUMP)
    main(parser.parse_args().fcidump)
