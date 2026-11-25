"""Train a bounded determinant pair on an explicit four-site Hubbard star.

Run: python -m examples.algorithms.fermion_hubbard_graph
"""

import argparse

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_graph
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable, number


def build_ansatz(
    angle: torch.Tensor, *, encoding: Encoding = "jordan_wigner"
) -> fq.Circuit:
    """Prepare the fixed physical pair in its explicitly selected binary basis."""
    if angle.ndim != 0:
        raise ValueError("the determinant-pair preparation requires a scalar angle")
    occupied: tuple[int, ...]
    flipped: tuple[int, ...]
    if encoding == "jordan_wigner":
        occupied, flipped = (3, 6), (6,)
    elif encoding == "parity":
        occupied, flipped = (3, 4, 5), (1, 2, 3, 4, 5)
    elif encoding == "bravyi_kitaev":
        occupied, flipped = (3, 6), (1, 3, 6)
    else:
        raise ValueError(f"unsupported encoding {encoding!r}")
    circuit = fq.Circuit(8, dtype=torch.complex128, device="cpu")
    for qubit in occupied:
        circuit.x(qubit)
    circuit.ry(0, theta=angle)
    for qubit in flipped:
        circuit.cx(0, qubit)
    return circuit


def energy(
    angle: torch.Tensor,
    observable: Observable,
    *,
    encoding: Encoding = "jordan_wigner",
) -> torch.Tensor:
    """Measure the matched determinant pair through the existing native runtime."""
    value: torch.Tensor = fq.run(
        build_ansatz(angle, encoding=encoding), outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


def main(encoding: Encoding = "jordan_wigner") -> None:
    print(
        "premise: explicit four-site star, fixed determinant pair, CPU "
        "single_device_fast_path, explicit encoding; only the circuit angle is "
        "trainable. No dense "
        "eigenstate preparation or general convergence/scalability claim.\n"
    )
    edges = ((0, 1), (0, 2), (0, 3))
    operator = fermi_hubbard_graph(4, edges, t=1.2, u=4, mu=0.2)
    observable = to_observable(
        encode_fermion(operator, n_modes=8, encoding=encoding).operator
    )
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    initial = energy(angle, observable, encoding=encoding)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable, encoding=encoding)
            - energy(angle - 1e-5, observable, encoding=encoding)
        ) / 2e-5
    optimizer = torch.optim.SGD([angle], lr=0.1)
    for _ in range(40):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable, encoding=encoding))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable, encoding=encoding)
        up = number(0) + number(2) + number(4) + number(6)
        down = number(1) + number(3) + number(5) + number(7)
        populations = [
            float(
                energy(
                    angle,
                    to_observable(
                        encode_fermion(op, n_modes=8, encoding=encoding).operator
                    ),
                    encoding=encoding,
                )
            )
            for op in (up, down)
        ]
    print(f"Encoding: {encoding}")
    print("Graph: 4 sites, 3 undirected edges")
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Gradient residual: {float((gradient - difference).abs()):.3e}")
    print(f"Alpha/beta populations: {populations[0]:.6f}, {populations[1]:.6f}")
    print("Analytic pair minimum: -1.6000000000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    main(parser.parse_args().encoding)
