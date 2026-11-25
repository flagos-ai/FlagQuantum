"""Train a three-site Hubbard ring with existing UCCSD and PyTorch.

Run: python -m examples.algorithms.fermion_hubbard_ring_training
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz, uccsd_excitations
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_ring
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, number


def energy(parameters: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure the existing two-electron UCCSD circuit in six full JW modes."""
    circuit = uccsd_ansatz(6, 2, parameters, dtype=torch.complex128, device="cpu")
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(hamiltonian)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    print(
        "premise: fixed three-site N_up=N_down=1 CPU training in full JW order; "
        "UCCSD angles are float32. This finite optimization run is not a general "
        "convergence, coefficient-gradient or scalability claim.\n"
    )
    encoded = encode_fermion(
        fermi_hubbard_ring(3, t=1, u=4), n_modes=6, encoding="jordan_wigner"
    )
    hamiltonian = to_observable(encoded.operator)
    count = uccsd_excitations(2, 6).parameter_count
    parameters = torch.linspace(
        -0.2, 0.3, count, dtype=torch.float32, requires_grad=True
    )
    initial_parameters = parameters.detach().clone()
    initial = energy(parameters, hamiltonian)
    gradient = torch.autograd.grad(initial, parameters)[0]
    # Resolve differences above the existing UCCSD float32 angle precision.
    epsilon = 1e-3
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    energy(parameters + shift, hamiltonian)
                    - energy(parameters - shift, hamiltonian)
                )
                / (2 * epsilon)
                for shift in epsilon * torch.eye(count, dtype=parameters.dtype)
            ]
        )
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    for _ in range(60):
        optimizer.zero_grad()
        torch.autograd.backward(energy(parameters, hamiltonian))
        optimizer.step()
    populations = [
        to_observable(
            encode_fermion(
                sum((number(mode) for mode in range(spin, 6, 2)), FermionOperator()),
                n_modes=6,
                encoding="jordan_wigner",
            ).operator
        )
        for spin in range(2)
    ]
    with torch.no_grad():
        final = energy(parameters, hamiltonian)
        circuit = uccsd_ansatz(6, 2, parameters, dtype=torch.complex128, device="cpu")
        result = fq.run(
            circuit, outputs=[fq.expectation(population) for population in populations]
        )
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Gradient norm: {float(gradient.norm()):.8f}")
    print(f"Gradient residual: {float((gradient - differences).abs().max()):.3e}")
    print(
        f"Parameter update norm: {float((parameters.detach() - initial_parameters).norm()):.8f}"
    )
    print(f"Final energy: {float(final):.10f}")
    print(f"Analytic N=2 ground energy: {(2 - 68**0.5) / 2:.10f}")
    print(f"Up particles: {float(result.expectation(0).reshape(())):.8f}")
    print(f"Down particles: {float(result.expectation(1).reshape(())):.8f}")


if __name__ == "__main__":
    main()
