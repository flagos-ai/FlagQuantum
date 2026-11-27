"""Opt-in linear correction of measured fixed-particle fermionic densities."""

from __future__ import annotations

import torch

from flagquantum.core.operators.pauli import count


def _lift(matrix: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    # Antisymmetric pair-space adjoint, with reversed annihilator axes.
    product = torch.einsum("ps,qr->pqrs", matrix, eye)
    return (
        product
        - product.transpose(0, 1)
        - product.transpose(2, 3)
        + product.permute(1, 0, 3, 2)
    )


def project_fixed_particle_densities(
    one_body: torch.Tensor, two_body: torch.Tensor, *, n_particles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project onto fixed-number linear constraints in physical mode order.

    Takes static CPU complex128 tensors on two through eight modes, with
    shapes (m, m) and (m, m, m, m) and 2 <= n_particles <= m. Returns new
    tensors in the same convention as the native density assemblers. The
    projection minimizes squared Frobenius distance in one-mode and
    independent pair coordinates with equal weight. It enforces Hermiticity,
    pair antisymmetry, both traces and the two-to-one contraction. It does
    not enforce positivity or N-representability.

    Raises:
        TypeError: An input is not a tensor or particle count is not an integer.
        ValueError: An input violates shape, precision, device, finiteness,
            static-data, particle-count, or numerical residual requirements.
    """
    if not isinstance(one_body, torch.Tensor) or not isinstance(two_body, torch.Tensor):
        raise TypeError("density inputs must be tensors")
    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one-body density must be square")
    width = one_body.shape[0]
    if not 2 <= width <= 8 or two_body.shape != (width,) * 4:
        raise ValueError("expected matching density shapes on two through eight modes")
    particles = count(n_particles, "n_particles", minimum=2)
    if particles > width:
        raise ValueError("n_particles cannot exceed the mode count")
    for density in (one_body, two_body):
        if (
            density.layout != torch.strided
            or density.device.type != "cpu"
            or density.dtype != torch.complex128
            or density.requires_grad
            or not bool(torch.isfinite(density).all())
        ):
            raise ValueError(
                "density inputs must be finite static dense CPU complex128 tensors"
            )

    gamma = 0.5 * (one_body + one_body.mH)
    pairs = 0.5 * (two_body - two_body.transpose(0, 1))
    pairs = 0.5 * (pairs - pairs.transpose(2, 3))
    pairs = 0.5 * (pairs + pairs.permute(3, 2, 1, 0).conj())
    eye = torch.eye(width, dtype=gamma.dtype, device=gamma.device)
    k = particles - 1
    gamma = gamma + ((particles - gamma.trace().real) / width) * eye
    pair_trace = torch.einsum("pqqp->", pairs).real
    pairs = pairs + ((particles * k - pair_trace) / (2 * width * (width - 1))) * _lift(
        eye, eye
    )

    # On traceless matrices C C* = (width - 2) I for the pair contraction C.
    residual = k * gamma - torch.einsum("pqqs->ps", pairs)
    correction = (residual - residual.trace() / width * eye) / (width - 2 + k**2)
    gamma = gamma - k * correction
    pairs = pairs + _lift(correction, eye)
    if not (
        bool(torch.isfinite(gamma).all())
        and bool(torch.isfinite(pairs).all())
        and gamma.trace().sub(particles).abs().item() <= 1e-10
        and torch.einsum("pqqp->", pairs).sub(particles * k).abs().item() <= 1e-10
        and (torch.einsum("pqqs->ps", pairs) - k * gamma).abs().max().item() <= 1e-10
    ):
        raise ValueError(
            "fixed-particle projection could not satisfy the numerical tolerance"
        )
    return gamma, pairs
