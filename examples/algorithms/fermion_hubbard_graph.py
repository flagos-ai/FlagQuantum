"""Train a bounded determinant pair on an explicit four-site Hubbard star.

Run: python -m examples.algorithms.fermion_hubbard_graph
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_graph
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable, number


def energy(angle: torch.Tensor, observable: Observable) -> torch.Tensor:
    """Measure a determinant pair with a down-spin spectator on site one."""
    circuit = fq.Circuit(8, dtype=torch.complex128, device="cpu").x(3).x(6)
    circuit.ry(0, theta=angle).cx(0, 6)
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    print(
        "premise: explicit four-site star, fixed determinant pair, CPU "
        "single_device_fast_path; only the circuit angle is trainable. No dense "
        "eigenstate preparation or general convergence/scalability claim.\n"
    )
    edges = ((0, 1), (0, 2), (0, 3))
    operator = fermi_hubbard_graph(4, edges, t=1.2, u=4, mu=0.2)
    observable = to_observable(encode_fermion(operator, n_modes=8).operator)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    initial = energy(angle, observable)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable) - energy(angle - 1e-5, observable)
        ) / 2e-5
    optimizer = torch.optim.SGD([angle], lr=0.1)
    for _ in range(40):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable)
        up = number(0) + number(2) + number(4) + number(6)
        down = number(1) + number(3) + number(5) + number(7)
        populations = [
            float(energy(angle, to_observable(encode_fermion(op, n_modes=8).operator)))
            for op in (up, down)
        ]
    print("Graph: 4 sites, 3 undirected edges")
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Gradient residual: {float((gradient - difference).abs()):.3e}")
    print(f"Alpha/beta populations: {populations[0]:.6f}, {populations[1]:.6f}")
    print("Analytic pair minimum: -1.6000000000")


if __name__ == "__main__":
    main()
