"""Train a fixed H2 determinant pair in an explicitly selected spin sector.

Run: python -m examples.algorithms.fermion_h2_sector_training
"""

import argparse
import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump
from flagquantum.observables import Observable, number
from flagquantum.observables.fermion import FermionOperator, FermionTerm

H2_FCIDUMP = Path(__file__).with_name("data") / "h2_sto3g.fcidump"


def build_ansatz(
    angle: torch.Tensor, *, encoding: Encoding = "jordan_wigner"
) -> fq.Circuit:
    """Prepare the same physical pair in its encoding-dependent reduced basis."""
    if angle.ndim != 0:
        raise ValueError("the determinant-pair preparation requires a scalar angle")
    circuit = fq.Circuit(2, dtype=torch.complex128, device="cpu")
    if encoding == "jordan_wigner":
        return circuit.ry(0, theta=angle).cx(0, 1)
    if encoding in ("parity", "bravyi_kitaev"):
        return circuit.ry(1, theta=angle)
    raise ValueError(f"unsupported encoding {encoding!r}")


def energy(
    angle: torch.Tensor,
    observable: Observable,
    *,
    encoding: Encoding = "jordan_wigner",
) -> torch.Tensor:
    """Measure an existing reduced observable through the native runtime."""
    value: torch.Tensor = fq.run(
        build_ansatz(angle, encoding=encoding), outputs=[fq.expectation(observable)]
    ).expectation()
    return value.reshape(())


def main(encoding: Encoding = "jordan_wigner") -> None:
    print(
        "premise: fixed two-orbital H2 FCIDUMP, explicit encoding and "
        "odd alpha/beta parities; CPU single_device_fast_path. Only the reduced "
        "circuit angle is trainable. The determinant pair is manually specified; "
        "discovery does not infer sectors or transform UCCSD. Static lifting and "
        "dense matrices are bounded inspection, not a scalable solver.\n"
    )
    problem = read_fcidump(H2_FCIDUMP, max_orbitals=2)
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4, encoding=encoding)
    operator = encoded.operator
    identity = FermionOperator((FermionTerm(1),))
    candidates = [
        encode_fermion(
            (identity - 2 * number(a)) * (identity - 2 * number(b)),
            n_modes=4,
            encoding=encoding,
        ).operator
        for a, b in ((0, 2), (1, 3))
    ]
    generators = discover_pauli_symmetries(operator, n_qubits=4, candidates=candidates)
    # Discovery may return products of candidates. Signs refer to those words,
    # using the explicitly supplied one-alpha/one-beta reference, not N alone.
    reference = encoded.reference_bits((1, 1, 0, 0))
    eigenvalues = [
        (-1) ** sum(reference[q] for q, _ in word.terms[0].factors)
        for word in generators
    ]
    reduced = reduce_stabilizers(
        operator, n_qubits=4, generators=generators, eigenvalues=eigenvalues
    )
    observable = to_observable(reduced.operator)
    up, down = number(0) + number(2), number(1) + number(3)
    populations = [
        to_observable(
            reduced.reduce_operator(
                encode_fermion(value, n_modes=4, encoding=encoding).operator
            )
        )
        for value in (up, down, up + down)
    ]
    angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    initial = energy(angle, observable, encoding=encoding)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable, encoding=encoding)
            - energy(angle - 1e-5, observable, encoding=encoding)
        ) / 2e-5
    optimizer = torch.optim.SGD([angle], lr=0.2)
    for _ in range(120):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable, encoding=encoding))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable, encoding=encoding)
        counts = [
            float(energy(angle, value, encoding=encoding)) for value in populations
        ]
        # Lifting remains static; training uses only reduced native expectations.
        lifted = lift_stabilizer_state(
            reduced, build_ansatz(angle, encoding=encoding).state().reshape(-1)
        )
        matrix = pauli_matrix(operator, n_qubits=4)
        full_energy = torch.vdot(lifted, matrix @ lifted).real
        first, second = (
            sum(bit << (3 - q) for q, bit in enumerate(encoded.reference_bits(bits)))
            for bits in ((1, 1, 0, 0), (0, 0, 1, 1))
        )
        a, c, b = (
            float(matrix[first, first].real),
            float(matrix[second, second].real),
            float(matrix[first, second].real),
        )
        minimum = (a + c) / 2 - math.hypot((a - c) / 2, b)
        sector_residual = max(
            float(
                (pauli_matrix(word, n_qubits=4) @ lifted - value * lifted).abs().max()
            )
            for word, value in zip(generators, eigenvalues, strict=True)
        )
    print(f"Encoding: {encoding}")
    print("Selected alpha/beta parity eigenvalues: -1, -1")
    print("Encoded generator eigenvalues: " + ", ".join(str(v) for v in eigenvalues))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    main(parser.parse_args().encoding)
