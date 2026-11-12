"""A native H2 controlled-time interferometer using existing circuit composition.

Run: python -m examples.algorithms.fermion_controlled_evolution
"""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.evolution import low_rank_evolution
from flagquantum.algorithms.fermion.low_rank import decompose_two_body
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.molecular import create_molecular_hamiltonian


def main() -> None:
    print(
        "premise: fixed H2 integrals in unreduced JW order; product-formula "
        "error is measured with a small dense reference, not bounded; "
        "one appended control retains the constant energy phase.\n"
    )
    molecule = from_native_molecule(
        create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    )
    factors = decompose_two_body(molecule.two_body)
    encoded = encode_fermion(
        fermion_hamiltonian(molecule), n_modes=4, encoding="jordan_wigner"
    )
    hamiltonian = pauli_matrix(encoded.operator, n_qubits=4)
    duration = 0.4
    reference = torch.matrix_exp(-1j * duration * hamiltonian)[12, 12]
    final_circuit = None
    for steps in (1, 2, 4):
        block = low_rank_evolution(
            molecule.one_body,
            factors,
            duration,
            constant=molecule.energy_offset,
            steps=steps,
            order=2,
        ).control(1, ctrl_qubits=4)
        circuit = fq.Circuit(5, dtype=torch.complex128).x(0).x(1).h(4)
        circuit.compose(block)
        result = fq.run(
            circuit, outputs=[fq.expectation(fq.X(4)), fq.expectation(fq.Y(4))]
        )
        real, imag = (
            float(torch.as_tensor(value).reshape(-1)[0])
            for value in result.expectations
        )
        residual = abs(complex(real, imag) - complex(reference))
        print(f"Steps {steps} overlap defect: {residual:.6e}")
        if steps == 4:
            print(f"Control X: {real:.10f}")
            print(f"Control Y: {imag:.10f}")
        final_circuit = circuit
    assert final_circuit is not None
    print(f"Register: 4 system + 1 control = {final_circuit.n_qubits} qubits")
    time = torch.tensor(duration, dtype=torch.float64, requires_grad=True)
    differentiable = fq.Circuit(5, dtype=torch.complex128).x(0).x(1).h(4)
    differentiable.compose(
        low_rank_evolution(
            molecule.one_body,
            factors,
            time,
            constant=molecule.energy_offset,
            steps=4,
        ).control(1, ctrl_qubits=4)
    )
    state = differentiable.state().reshape(-1)
    overlap = 2 * (state[::2].conj() * state[1::2]).sum()
    gradient = torch.autograd.grad(overlap.real, time)[0]
    print(f"Control X time derivative: {float(gradient):.10f}")
    print("No phase alignment or energy-offset removal is used.")


if __name__ == "__main__":
    main()
