"""Train a fixed H2 determinant pair in an explicitly selected spin sector.

Run: python -m examples.algorithms.fermion_h2_sector_training
"""

import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.core.algebra.pauli import pauli
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import Observable, number

H2_FCIDUMP = Path(__file__).with_name("data") / "h2_sto3g.fcidump"


def build_ansatz(angle: torch.Tensor) -> fq.Circuit:
    """Prepare the reduced determinant pair with one scalar angle."""
    if angle.ndim != 0:
        raise ValueError("the determinant-pair preparation requires a scalar angle")
    return (
        fq.Circuit(2, dtype=torch.complex128, device="cpu").ry(0, theta=angle).cx(0, 1)
    )


def energy(angle: torch.Tensor, observable: Observable) -> torch.Tensor:
    """Measure an existing reduced observable through the native runtime."""
    value: torch.Tensor = fq.run(
        build_ansatz(angle), outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    print(
        "premise: fixed two-orbital H2 FCIDUMP, full JW order and explicit "
        "odd alpha/beta parities; CPU single_device_fast_path. Only the reduced "
        "circuit angle is trainable. The determinant pair is manually specified; "
        "discovery does not infer sectors or transform UCCSD. Static lifting and "
        "dense matrices are bounded inspection, not a scalable solver.\n"
    )
    problem = read_fcidump(H2_FCIDUMP, max_orbitals=2)
    operator = encode_fermion(fermion_hamiltonian(problem), n_modes=4).operator
    candidates = [
        pauli(factors=((0, "z"), (2, "z"))),
        pauli(factors=((1, "z"), (3, "z"))),
    ]
    generators = discover_pauli_symmetries(operator, n_qubits=4, candidates=candidates)
    reduced = reduce_stabilizers(
        operator, n_qubits=4, generators=generators, eigenvalues=[-1, -1]
    )
    observable = to_observable(reduced.operator)
    up, down = number(0) + number(2), number(1) + number(3)
    populations = [
        to_observable(
            reduced.reduce_operator(encode_fermion(value, n_modes=4).operator)
        )
        for value in (up, down, up + down)
    ]
    angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    initial = energy(angle, observable)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable) - energy(angle - 1e-5, observable)
        ) / 2e-5
    optimizer = torch.optim.SGD([angle], lr=0.2)
    for _ in range(120):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable)
        counts = [float(energy(angle, value)) for value in populations]
        # Lifting remains static; training uses only reduced native expectations.
        lifted = lift_stabilizer_state(reduced, build_ansatz(angle).state().reshape(-1))
        matrix = pauli_matrix(operator, n_qubits=4)
        full_energy = torch.vdot(lifted, matrix @ lifted).real
        a, c, b = (
            float(matrix[12, 12].real),
            float(matrix[3, 3].real),
            float(matrix[12, 3].real),
        )
        minimum = (a + c) / 2 - math.hypot((a - c) / 2, b)
        sector_residual = max(
            float((pauli_matrix(word, n_qubits=4) @ lifted + lifted).abs().max())
            for word in generators
        )
    print("Selected alpha/beta parity eigenvalues: -1, -1")
    print(f"Register: 4 -> {reduced.n_qubits} qubits")
    print(f"Initial energy: {float(initial.detach()):.10f} Ha")
    print(f"Gradient residual: {float((gradient - difference).abs()):.3e}")
    print(f"Parameter update: {float((angle.detach() - before).abs()):.10f}")
    print(f"Final energy: {float(final):.10f} Ha")
    print(f"Analytic determinant-pair minimum: {minimum:.10f} Ha")
    print(f"Alpha particles: {counts[0]:.8f}")
    print(f"Beta particles: {counts[1]:.8f}")
    print(f"Particles: {counts[2]:.8f}")
    print(f"Lifted energy residual: {float((full_energy - final).abs()):.3e}")
    print(f"Joint-sector residual: {sector_residual:.3e}")


if __name__ == "__main__":
    main()
