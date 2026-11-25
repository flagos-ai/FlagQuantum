"""Measure a complex fermionic 1-RDM and reconstruct one-body energy on CPU.

Run: python -m examples.algorithms.fermion_one_body_density --encoding parity
"""

from __future__ import annotations

import argparse

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.density import (
    one_body_density,
    one_body_density_observables,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import annihilate, create, number


def build_circuit(angle: torch.Tensor, *, encoding: Encoding) -> fq.Circuit:
    """Prepare cos(angle/2)|10> + i sin(angle/2)|01> in the chosen basis."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {encoding!r}")
    circuit = fq.Circuit(2, dtype=torch.complex128, device="cpu")
    circuit.x(0).ry(1, theta=angle).cx(1, 0).s(1)
    if encoding != "jordan_wigner":
        # At width two, prefix parity and Fenwick BK both store (n0, n0 xor n1).
        circuit.cx(0, 1)
    return circuit


def main(encoding: Encoding = "jordan_wigner") -> None:
    angle = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    circuit = build_circuit(angle, encoding=encoding)
    observables = one_body_density_observables(n_modes=2, encoding=encoding)
    hop = create(0) * annihilate(1)
    operator = 0.7 * number(0) - 0.2 * number(1) + 0.3j * hop - 0.3j * hop.dagger()
    direct_observable = to_observable(
        encode_fermion(operator, n_modes=2, encoding=encoding).operator
    )
    result = fq.run(
        circuit,
        outputs=[fq.expectation(o) for o in (*observables, direct_observable)],
    )
    gamma = one_body_density(result.expectations[:-1], n_modes=2)
    h = torch.tensor([[0.7, 0.3j], [-0.3j, -0.2]], dtype=torch.complex128)
    reconstructed = (h * gamma).sum()
    torch.testing.assert_close(
        reconstructed.imag, angle.new_zeros(()), atol=1e-12, rtol=0
    )
    direct = result.expectations[-1].reshape(())
    torch.testing.assert_close(reconstructed.real, direct, atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(reconstructed.real, angle)[0]
    print(
        "premise: CPU single_device_fast_path; exact expectations on a full register. "
        "Statevector storage remains exponential; no sampled, hardware or scalability claim.\n"
    )
    print(f"Encoding: {encoding}")
    print(f"Particle number: {float(gamma.trace().real.detach()):.10f}")
    print(f"One-body energy: {float(reconstructed.real.detach()):.10f}")
    print(f"Angle gradient: {float(gradient):.10f}")
    print(
        f"Direct energy residual: {float((reconstructed.real - direct).abs().detach()):.3e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    main(parser.parse_args().encoding)
