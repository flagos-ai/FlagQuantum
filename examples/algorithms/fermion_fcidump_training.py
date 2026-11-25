"""Train the existing UCCSD circuit from a bounded two-orbital FCIDUMP.

Run: python -m examples.algorithms.fermion_fcidump_training [FCIDUMP] --encoding parity
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import (
    ENCODINGS,
    Encoding,
    encode_fermion,
    encode_occupation,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, number

DEFAULT_FCIDUMP = Path(__file__).with_name("data") / "h2_sto3g.fcidump"


def _encoded_ansatz(parameters: torch.Tensor, encoding: Encoding) -> fq.Circuit:
    circuit = uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
    if encoding == "jordan_wigner":
        return circuit
    # Each encoded unit occupation gives a column of the binary basis map.
    # Visit targets high to low so every lower control is still an occupation bit.
    columns = tuple(
        encode_occupation(tuple(int(q == control) for q in range(4)), encoding=encoding)
        for control in range(4)
    )
    for target in range(3, 0, -1):
        for control in range(target):
            if columns[control][target]:
                circuit.cx(control, target)
    return circuit


def energy(
    parameters: torch.Tensor,
    hamiltonian: Observable,
    *,
    encoding: Encoding = "jordan_wigner",
) -> torch.Tensor:
    """Measure UCCSD against a Hamiltonian in the caller's matching encoding."""
    circuit = _encoded_ansatz(parameters, encoding)
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(hamiltonian)]
    ).expectation()
    return value.reshape(())


def train_adam(
    parameters: torch.Tensor,
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 160,
) -> None:
    """Update the supplied angles with the shared H2 training loop."""
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        torch.autograd.backward(evaluate(parameters))
        optimizer.step()


def main(path: Path = DEFAULT_FCIDUMP, *, encoding: Encoding = "jordan_wigner") -> None:
    print(
        "premise: two-orbital closed-shell CPU single_device_fast_path on the full register; "
        "UCCSD angles are float32 and the dense N=2 reference has exponential "
        "inspection cost, not a general convergence or scalability claim.\n"
    )
    problem = read_fcidump(path, max_orbitals=2)
    if problem.n_orbitals != 2 or problem.n_electrons != 2:
        raise ValueError(
            "this training example requires two orbitals and two electrons"
        )
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4, encoding=encoding)
    observable = to_observable(encoded.operator)
    matrix = pauli_matrix(encoded.operator, n_qubits=4)
    sector = []
    for index in range(16):
        if index.bit_count() == 2:
            occupation = tuple((index >> (3 - q)) & 1 for q in range(4))
            bits = encoded.reference_bits(occupation)
            sector.append(sum(bit << (3 - q) for q, bit in enumerate(bits)))
    reference = torch.linalg.eigvalsh(matrix[sector][:, sector])[0]
    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    initial_parameters = parameters.detach().clone()
    initial = energy(parameters, observable, encoding=encoding)
    gradient = torch.autograd.grad(initial, parameters)[0]
    # The existing UCCSD boundary normalizes angles to float32, including when
    # supplied as float64. Resolve differences above its quantization scale.
    epsilon = 1e-3
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    energy(parameters + shift, observable, encoding=encoding)
                    - energy(parameters - shift, observable, encoding=encoding)
                )
                / (2 * epsilon)
                for shift in epsilon * torch.eye(3, dtype=parameters.dtype)
            ]
        )
        determinant = energy(
            torch.zeros_like(parameters), observable, encoding=encoding
        )
    train_adam(parameters, lambda angles: energy(angles, observable, encoding=encoding))
    particles = sum((number(mode) for mode in range(4)), FermionOperator())
    population = to_observable(
        encode_fermion(particles, n_modes=4, encoding=encoding).operator
    )
    with torch.no_grad():
        final = energy(parameters, observable, encoding=encoding)
        circuit = _encoded_ansatz(parameters, encoding)
        count = fq.run(circuit, outputs=[fq.expectation(population)]).expectation()
    print(f"Encoding: {encoding}")
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
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    args = parser.parse_args()
    main(args.fcidump, encoding=args.encoding)
