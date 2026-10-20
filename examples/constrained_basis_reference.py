"""Labelled-basis evolution using existing operator, plan and gradient APIs.

The caller validates each reduced model and owns its physical interpretation.
Plans retain ordered literal basis labels, not certified number/fusion rules.
Ising conventions: Nayak et al., arXiv:0707.1889, section II.A and equation (50).
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create


def main() -> None:
    dtype = torch.complex128
    # One photon in two modes; truncation to levels 0 and 1 is exact here.
    operator = create(0) * annihilate(1) + create(1) * annihilate(0)
    full_hamiltonian = operator.to_matrix({0: 2, 1: 2})
    labels = ((0, 1), (1, 0))
    indices = [2 * first + second for first, second in labels]
    embedding = torch.eye(4, dtype=dtype)[:, indices]
    complement = torch.eye(4, dtype=dtype) - embedding @ embedding.mH
    assert torch.max(torch.abs(complement @ full_hamiltonian @ embedding)) == 0
    reduced = embedding.mH @ full_hamiltonian @ embedding
    theta = torch.tensor(math.pi / 4, dtype=torch.float64, requires_grad=True)
    result = fql.run(
        theta * reduced,
        [1.0, 0.0],
        [0.0, 1.0],
        basis_labels=labels,
        method="krylov-exponential",
        options=fq.ExecutionOptions(require_gradients=True),
    )
    probability = result.populations[-1, labels.index((1, 0))]
    (gradient,) = torch.autograd.grad(probability, theta)
    print(f"One-excitation transfer: {probability.item():.8f}")
    print(f"Control derivative: {gradient.item():.8f}")

    # Caller-owned fusion convention: four sigma anyons, total 1, left tree,
    # paths (1, sigma, 1), (psi, sigma, 1). No anyon model is created by fql.
    fusion_labels = (("1", "sigma", "1"), ("psi", "sigma", "1"))
    change_basis = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype) / math.sqrt(2)
    r = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=dtype) * math.pi / 8))
    exchange = change_basis.mH @ r @ change_basis
    # This Hamiltonian is a numerical logarithm witness of that one unitary.
    h = math.pi / 4 * reduced - math.pi / 8 * torch.eye(2, dtype=dtype)
    fusion_plan = fql.plan(
        h,
        [1.0, 0.0],
        [0.0, 1.0],
        basis_labels=fusion_labels,
        method="krylov-exponential",
        return_density_matrices=True,
    )
    encoded = fql.run(fusion_plan)
    expected_state = exchange[:, 0]
    expected_density = torch.outer(expected_state, expected_state.conj())
    torch.testing.assert_close(
        encoded.density_matrices[-1], expected_density, atol=1e-12, rtol=1e-12
    )
    braid_error = torch.max(torch.abs(r @ exchange @ r - exchange @ r @ exchange))
    print(f"Fusion-channel probability: {encoded.populations[-1, 1].item():.8f}")
    print(f"Braid relation verified: {bool(braid_error < 1e-12)}")
    has_labels = fusion_plan.basis_labels == fusion_labels
    assert has_labels
    print("Plan retains declared basis labels:", has_labels)


if __name__ == "__main__":
    main()
