"""Check fixed-particle projection against an independent pair-space solve."""

from itertools import combinations

import pytest
import torch

from flagquantum.algorithms.fermion.density_projection import (
    project_fixed_particle_densities,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("width", "particles"), [(2, 2), (4, 2), (4, 3), (8, 8)])
def test_projection_matches_dense_linear_reference(width: int, particles: int) -> None:
    generator = torch.Generator().manual_seed(819)
    gamma = torch.randn(width, width, dtype=torch.complex128, generator=generator)
    density = torch.randn((width,) * 4, dtype=torch.complex128, generator=generator)
    original = gamma.clone(), density.clone()
    one, two = project_fixed_particle_densities(gamma, density, n_particles=particles)
    torch.testing.assert_close(gamma, original[0], atol=0, rtol=0)
    torch.testing.assert_close(density, original[1], atol=0, rtol=0)
    torch.testing.assert_close(one, one.mH, atol=1e-12, rtol=0)
    torch.testing.assert_close(two, -two.transpose(0, 1), atol=1e-12, rtol=0)
    torch.testing.assert_close(two, -two.transpose(2, 3), atol=1e-12, rtol=0)
    torch.testing.assert_close(two, two.permute(3, 2, 1, 0).conj(), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        one.trace(), one.new_tensor(particles), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        torch.einsum("pqqs->ps", two), (particles - 1) * one, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        torch.einsum("pqqp->", two),
        two.new_tensor(particles * (particles - 1)),
        atol=1e-12,
        rtol=0,
    )
    again = project_fixed_particle_densities(one, two, n_particles=particles)
    torch.testing.assert_close(again, (one, two), atol=1e-12, rtol=0)
    # The independent least-norm reference is only needed at small widths.
    if width > 4:
        return
    pairs = list(combinations(range(width), 2))
    pair_count = len(pairs)
    contract = torch.zeros((width**2, pair_count**2), dtype=torch.complex128)
    measured = torch.empty((pair_count, pair_count), dtype=torch.complex128)
    result = torch.empty_like(measured)
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            measured[i, j] = (
                density[p, q, s, r]
                - density[q, p, s, r]
                - density[p, q, r, s]
                + density[q, p, r, s]
            ) / 4
            result[i, j] = two[p, q, s, r]
            for a, b, left_sign in ((p, q, 1), (q, p, -1)):
                for c, d, right_sign in ((r, s, 1), (s, r, -1)):
                    if b == d:
                        contract[a * width + c, i * pair_count + j] += (
                            left_sign * right_sign
                        )
    trace = torch.eye(width, dtype=one.dtype).flatten()
    constraints = torch.cat(
        (
            torch.cat((trace, torch.zeros(pair_count**2, dtype=one.dtype)))[None, :],
            torch.cat(
                (-(particles - 1) * torch.eye(width**2, dtype=one.dtype), contract),
                dim=1,
            ),
        )
    )
    target = torch.cat((one.new_tensor([particles]), one.new_zeros(width**2)))
    initial = torch.cat(
        ((gamma + gamma.mH).flatten() / 2, (measured + measured.mH).flatten() / 2)
    )
    expected = initial + torch.linalg.pinv(constraints) @ (
        target - constraints @ initial
    )
    torch.testing.assert_close(
        torch.cat((one.flatten(), result.flatten())), expected, atol=1e-12, rtol=0
    )


def test_projection_refuses_unsupported_inputs() -> None:
    one = torch.eye(2, dtype=torch.complex128)
    two = torch.zeros((2,) * 4, dtype=one.dtype)
    for particles in (0, 1, 3):
        with pytest.raises(ValueError, match="n_particles"):
            project_fixed_particle_densities(one, two, n_particles=particles)
    with pytest.raises(ValueError, match="matching density shapes"):
        project_fixed_particle_densities(one, two[..., :1], n_particles=2)
    for invalid in (
        one.to(torch.complex64),
        one.clone().requires_grad_(),
        one * float("nan"),
    ):
        with pytest.raises(ValueError, match="finite static dense"):
            project_fixed_particle_densities(invalid, two, n_particles=2)
    with pytest.raises(ValueError, match="numerical tolerance"):
        project_fixed_particle_densities(one * 1e308, two, n_particles=2)
