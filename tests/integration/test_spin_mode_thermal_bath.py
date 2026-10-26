"""Frozen scalar bath, finite boundary and independent spin-mode acceptance."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

pytestmark = pytest.mark.integration


def reference_initial(
    cutoff: int, *, coherent: bool = False, ratio: float = 0.2
) -> torch.Tensor:
    spin = torch.tensor([1, 1j] if coherent else [0, 1], dtype=torch.complex128)
    if coherent:
        spin = spin / math.sqrt(2)
    weights = torch.tensor([ratio**n for n in range(cutoff)], dtype=torch.float64)
    weights /= weights.sum()
    return torch.kron(
        torch.outer(spin, spin.conj()), torch.diag(weights).to(torch.complex128)
    )


def reference_operators(
    cutoff: int, coupling: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = 2 * cutoff
    h = torch.zeros((dim, dim), dtype=torch.complex128)
    down = torch.zeros_like(h)
    for n in range(cutoff - 1):
        h[cutoff + n, n + 1] = h[n + 1, cutoff + n] = coupling * math.sqrt(n + 1)
    for spin in range(2):
        for n in range(1, cutoff):
            down[spin * cutoff + n - 1, spin * cutoff + n] = math.sqrt(n)
    return h, down, down.mH


def reference_generator(
    cutoff: int, coupling: float, rates: tuple[float, float] = (0.3, 0.1)
) -> torch.Tensor:
    """Columns are the scalar master equation applied to elementary E_ab."""
    h, down, up = reference_operators(cutoff, coupling)
    dim = 2 * cutoff
    columns = []
    for a in range(dim):
        for b in range(dim):
            e = torch.zeros_like(h)
            e[a, b] = 1
            derivative = -1j * (h @ e - e @ h)
            for jump, rate in zip((down, up), rates, strict=True):
                occupation = jump.mH @ jump
                derivative += rate * (
                    jump @ e @ jump.mH - (occupation @ e + e @ occupation) / 2
                )
            columns.append(derivative.reshape(dim**2))
    return torch.stack(columns, dim=1)


def reference_states(
    cutoff: int,
    coupling: float,
    *,
    coherent: bool = False,
    rates: tuple[float, float] = (0.3, 0.1),
) -> torch.Tensor:
    initial = reference_initial(cutoff, coherent=coherent)
    generator = reference_generator(cutoff, coupling, rates)
    return torch.stack(
        [
            (torch.matrix_exp(time * generator) @ initial.reshape(-1)).reshape(
                2 * cutoff, 2 * cutoff
            )
            for time in (0.0, 0.35, 0.7)
        ]
    )


def reference_birth_death(
    cutoff: int, rates: tuple[float, float] = (0.3, 0.1)
) -> torch.Tensor:
    generator = torch.zeros((cutoff, cutoff), dtype=torch.float64)
    for n in range(cutoff):
        if n > 0:
            generator[n - 1, n] = rates[0] * n
            generator[n, n] -= rates[0] * n
        if n < cutoff - 1:
            generator[n + 1, n] = rates[1] * (n + 1)
            generator[n, n] -= rates[1] * (n + 1)
    return generator


def reference_moments(rows: torch.Tensor, cutoff: int) -> torch.Tensor:
    populations = rows.diagonal(dim1=-2, dim2=-1).real.reshape(-1, 2, cutoff)
    return torch.stack(
        (
            populations[:, 1].sum(-1),
            (populations.sum(1) * torch.arange(cutoff, dtype=torch.float64)).sum(-1),
            populations[:, :, -1].sum(-1),
        ),
        dim=-1,
    )


@pytest.mark.parametrize("cutoff", [3, 5, 7, 9, 11])
def test_reference_bath_stationarity_mean_bound_and_finite_moment(cutoff: int) -> None:
    q = 0.2
    weights = torch.tensor([q**n for n in range(cutoff)], dtype=torch.float64)
    weights /= weights.sum()
    bath = reference_birth_death(cutoff)
    stationary = torch.tensor(
        [(1 / 3) ** n for n in range(cutoff)], dtype=torch.float64
    )
    stationary /= stationary.sum()
    torch.testing.assert_close(
        bath @ stationary, torch.zeros(cutoff, dtype=torch.float64), atol=1e-12, rtol=0
    )
    delta0 = cutoff * q**cutoff / (1 - q**cutoff)
    augmented = torch.zeros((cutoff + 1, cutoff + 1), dtype=torch.float64)
    augmented[:cutoff, :cutoff] = bath
    augmented[cutoff, cutoff - 1] = 0.1 * cutoff
    augmented[cutoff, cutoff] = -0.2
    for time in (0.0, 0.35, 0.7):
        population = torch.matrix_exp(time * bath) @ weights
        mean = float(population @ torch.arange(cutoff, dtype=torch.float64))
        deficit = 0.5 - 0.25 * math.exp(-0.2 * time) - mean
        bound = delta0 + 0.1 * cutoff * time * (1 - 0.8 * math.exp(-0.1 * time)) ** (
            cutoff - 1
        )
        assert -1e-10 <= deficit <= bound + 1e-10
        integral_deficit = (
            torch.matrix_exp(time * augmented)
            @ torch.cat((weights, torch.tensor([delta0], dtype=torch.float64)))
        )[-1]
        assert abs(deficit - float(integral_deficit)) <= 1e-10
    number = torch.diag(
        torch.tensor(
            [n for _ in range(2) for n in range(cutoff)], dtype=torch.complex128
        )
    )
    spin = torch.diag(torch.tensor([0] * cutoff + [1] * cutoff, dtype=torch.complex128))
    top = torch.diag(
        torch.tensor(
            [int(n == cutoff - 1) for _ in range(2) for n in range(cutoff)],
            dtype=torch.complex128,
        )
    )
    expected = -0.2 * number + 0.1 * (
        torch.eye(2 * cutoff, dtype=torch.complex128) - cutoff * top
    )
    adjoint = (
        reference_generator(cutoff, 0.6).mH @ (number + spin).reshape(-1)
    ).reshape(2 * cutoff, 2 * cutoff)
    torch.testing.assert_close(adjoint, expected, atol=1e-12, rtol=0)
    assert float(abs(0.1 * cutoff * top).max()) > 0


@pytest.mark.parametrize("cutoff", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("coherent", [False, True])
def test_coupled_full_density_and_existing_axes(cutoff: int, coherent: bool) -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    plan = thermal_bath_plan(
        cutoff, torch.tensor(0.6, dtype=torch.float64), coherent=coherent
    )
    result = fql.run(plan)
    assert result.density_matrices is not None
    rows = result.density_matrices
    assert rows.shape == (3, 2 * cutoff, 2 * cutoff)
    assert plan.dimensions == (2, cutoff)
    torch.testing.assert_close(
        rows, reference_states(cutoff, 0.6, coherent=coherent), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        rows.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.ones(3, dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )
    assert float(abs(rows - rows.mH).max()) <= 1e-10
    assert float(torch.linalg.eigvalsh(rows).min()) >= -1e-10
    moments = reference_moments(rows, cutoff)
    for index, name in enumerate(("excited", "mode", "top")):
        torch.testing.assert_close(
            result.expectation(name), moments[:, index], atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("cutoff", [3, 5, 7, 9, 11])
def test_bath_only_population_spin_coherence_and_zero_rate(cutoff: int) -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    initial = reference_initial(cutoff, coherent=True)
    spin = torch.tensor([[0.5, -0.5j], [0.5j, 0.5]], dtype=torch.complex128)
    weights = initial.diagonal().real[:cutoff] * 2
    result = fql.run(
        thermal_bath_plan(cutoff, torch.tensor(0.0, dtype=torch.float64), coherent=True)
    )
    assert result.density_matrices is not None
    for index, time in enumerate((0.0, 0.35, 0.7)):
        population = torch.matrix_exp(time * reference_birth_death(cutoff)) @ weights
        expected = torch.kron(spin, torch.diag(population).to(torch.complex128))
        torch.testing.assert_close(
            result.density_matrices[index], expected, atol=1e-10, rtol=0
        )
    zero = fql.run(
        thermal_bath_plan(
            cutoff,
            torch.tensor(0.0, dtype=torch.float64),
            coherent=True,
            rates=(0.0, 0.0),
        )
    )
    assert zero.density_matrices is not None
    torch.testing.assert_close(
        zero.density_matrices, initial.expand(3, -1, -1), atol=1e-10, rtol=0
    )


def test_zero_bath_retains_lossless_rabi_reference() -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_convergence import reference_density
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    for cutoff in (3, 5, 7, 9, 11):
        result = fql.run(
            thermal_bath_plan(
                cutoff, torch.tensor(0.6, dtype=torch.float64), rates=(0.0, 0.0)
            )
        )
        assert result.density_matrices is not None
        for index, time in enumerate((0.0, 0.35, 0.7)):
            torch.testing.assert_close(
                result.density_matrices[index],
                reference_density(cutoff, 0.6, time),
                atol=1e-10,
                rtol=0,
            )


def test_cutoff_stability_is_separate_from_finite_density_error() -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    observations = {}
    for cutoff in (3, 5, 7, 9, 11):
        result = fql.run(
            thermal_bath_plan(cutoff, torch.tensor(0.6, dtype=torch.float64))
        )
        observations[cutoff] = torch.stack(
            [result.expectation(n) for n in ("excited", "mode", "top")], dim=-1
        )
    differences = abs(observations[9] - observations[11])
    assert float(differences[:, 0].max()) < 5e-5
    assert float(differences[:, 1].max()) < 1e-4
    assert float(observations[11][:, 2].max()) < 1e-5


def test_fixed_cutoff_integration_refinement() -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    reference = reference_states(3, 0.6)
    errors = []
    for steps in (8, 16, 32, 64):
        result = fql.run(
            thermal_bath_plan(
                3,
                torch.tensor(0.6, dtype=torch.float64),
                steps=steps,
                method="runge-kutta",
            )
        )
        assert result.density_matrices is not None
        errors.append(
            float(abs(result.density_matrices[:: steps // 2] - reference).max())
        )
    assert all(a / b > 10 for a, b in pairwise(errors))
    assert errors[-1] < 1e-6


def test_gradient_fitting_and_detached_replay() -> None:
    import flagquantum.lindblad as fql
    from examples.spin_mode_thermal_bath import thermal_bath_plan

    coupling = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    result = fql.run(thermal_bath_plan(5, coupling))
    (gradient,) = torch.autograd.grad(result.expectation("excited")[-1], coupling)
    step = 1e-4
    values = [
        reference_moments(reference_states(5, 0.6 + multiple * step), 5)[-1, 0]
        for multiple in (-2, -1, 1, 2)
    ]
    expected = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
    assert abs(float(gradient - expected)) <= 1e-9
    target = reference_moments(reference_states(5, 0.6), 5)[-1, 0]
    fitted = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
    optimizer = torch.optim.Adam([fitted], lr=0.04)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        prediction = fql.run(thermal_bath_plan(5, fitted)).expectation("excited")[-1]
        loss = (prediction - target).square()
        torch.autograd.backward(loss)
        assert fitted.grad is not None and bool(torch.isfinite(fitted.grad))
        losses.append(float(loss.detach()))
        optimizer.step()
    assert all(math.isfinite(loss) for loss in losses)
    plan = thermal_bath_plan(5, fitted.detach())
    final = fql.run(plan)
    assert (
        float((final.expectation("excited")[-1] - target).square()) <= 0.1 * losses[0]
    )
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert final.density_matrices is not None and replay.density_matrices is not None
    assert torch.equal(final.density_matrices, replay.density_matrices)
    for name in ("excited", "mode", "top"):
        assert torch.equal(final.expectation(name), replay.expectation(name))
    torch.testing.assert_close(
        final.density_matrices,
        reference_states(5, float(fitted.detach())),
        atol=1e-10,
        rtol=0,
    )
