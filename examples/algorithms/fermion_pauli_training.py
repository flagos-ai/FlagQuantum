"""Train one circuit angle in an explicitly selected mixed-axis Pauli sector.

Run: python -m examples.algorithms.fermion_pauli_training
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.core.algebra.pauli import pauli
from flagquantum.observables import Observable


def build_ansatz(angle: torch.Tensor) -> fq.Circuit:
    """Prepare the reduced qubit with one scalar circuit angle."""
    if angle.ndim != 0:
        raise ValueError("the reduced preparation requires a scalar angle")
    return fq.Circuit(1, dtype=torch.complex128, device="cpu").ry(0, theta=angle)


def energy(angle: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure a reduced observable through the existing native runtime."""
    value: torch.Tensor = fq.run(
        build_ansatz(angle), outputs=[fq.expectation(hamiltonian)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    print(
        "premise: fixed three-qubit encoded Pauli model and explicit +1/-1 "
        "joint sector; CPU single_device_fast_path. Only the circuit angle is "
        "trainable. Discovery does not select a ground sector or particle "
        "number. Static lifting is bounded inspection, not a general "
        "convergence, coefficient-gradient or scalability claim.\n"
    )
    y0 = pauli(factors=((0, "y"),))
    x1z2 = pauli(factors=((1, "x"), (2, "z")))
    operator = (
        pauli(factors=((2, "z"),))
        + 0.4 * pauli(factors=((1, "z"), (2, "x")))
        + 0.2 * y0
        + 0.3 * x1z2
    )
    generators = discover_pauli_symmetries(operator, n_qubits=3)
    reduced = reduce_stabilizers(
        operator, n_qubits=3, generators=generators, eigenvalues=[1, -1]
    )
    observable = to_observable(reduced.operator)
    angle = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    initial = energy(angle, observable)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable) - energy(angle - 1e-5, observable)
        ) / 2e-5
    optimizer = torch.optim.SGD([angle], lr=0.1)
    for _ in range(120):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable)
        # Static lifting inspects the final state; it is not an autograd path.
        lifted = lift_stabilizer_state(reduced, build_ansatz(angle).state().reshape(-1))
        full_energy = (lifted.conj() @ pauli_matrix(operator, n_qubits=3) @ lifted).real
        sector_residual = max(
            float((pauli_matrix(word, n_qubits=3) @ lifted - sign * lifted).abs().max())
            for word, sign in zip(generators, (1, -1), strict=True)
        )
    print("Selected Y0 and X1Z2 eigenvalues: +1, -1")
    print(f"Register: 3 -> {reduced.n_qubits} qubits")
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Gradient residual: {float((gradient - difference).abs()):.3e}")
    print(f"Parameter update: {float((angle.detach() - before).abs()):.10f}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Analytic selected-sector minimum: {-0.1 - 1.16**0.5:.10f}")
    print(f"Lifted energy residual: {float((full_energy - final).abs()):.3e}")
    print(f"Joint-sector residual: {sector_residual:.3e}")


if __name__ == "__main__":
    main()
