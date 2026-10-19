"""Bounded boson models share existing dynamics, plans, and gradients."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables import boson
from flagquantum.simulation.lindblad import (
    EvolutionValidationError,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)

pytestmark = pytest.mark.unit


def _interference() -> boson.BosonOperator:
    return 0.7 * (
        boson.create(0) * boson.annihilate(1) + boson.create(1) * boson.annihilate(0)
    )


def _pair(levels: int) -> torch.Tensor:
    state = torch.zeros(levels * levels, dtype=torch.complex128)
    state[levels + 1] = 1  # One photon in each mode, subsystem 0 most significant.
    return state


@pytest.mark.integration
@pytest.mark.parametrize(
    "method", ["runge-kutta", "crank-nicolson", "krylov-exponential"]
)
def test_two_photon_interference_and_uniform_loss(method: str) -> None:
    final_time = math.pi / (4 * 0.7)
    times = torch.linspace(0.0, final_time, 101, dtype=torch.float64)
    rate = 0.2
    result = fql.run(
        _interference(),
        _pair(3),
        times,
        dimensions=(3, 3),
        collapse_operators=[(boson.annihilate(0), rate), (boson.annihilate(1), rate)],
        method=method,
    )
    # Linear optics predicts P11=cos(2*g*t)^2 and each bunched outcome
    # sin(2*g*t)^2/2. Independent loss keeps each photon with probability p.
    survival = torch.exp(-rate * times)
    coincidence = survival.square() * torch.cos(2 * 0.7 * times).square()
    bunching = survival.square() * torch.sin(2 * 0.7 * times).square() / 2
    tolerance = 3e-5 if method == "crank-nicolson" else 2e-8
    torch.testing.assert_close(
        result.populations[:, 4], coincidence, atol=tolerance, rtol=0
    )
    torch.testing.assert_close(
        result.populations[:, 2], bunching, atol=tolerance, rtol=0
    )
    torch.testing.assert_close(
        result.populations[:, 6], bunching, atol=tolerance, rtol=0
    )
    torch.testing.assert_close(
        result.populations[:, 0], (1 - survival).square(), atol=tolerance, rtol=0
    )
    torch.testing.assert_close(
        result.populations[:, 1] + result.populations[:, 3],
        2 * survival * (1 - survival),
        atol=tolerance,
        rtol=0,
    )
    torch.testing.assert_close(
        result.populations.sum(-1), torch.ones_like(times), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("dimensions", [(3, 4), (2, 2)])
def test_direct_and_materialized_inputs_seal_the_same_plan(
    dimensions: tuple[int, ...],
) -> None:
    levels = dict(enumerate(dimensions))
    h = _interference()
    loss = boson.annihilate(1)
    state = torch.zeros(math.prod(dimensions), dtype=torch.complex128)
    state[dimensions[1]] = 1
    direct = fql.plan(
        h, state, [0.0, 0.2], dimensions=dimensions, collapse_operators=[(loss, 0.3)]
    )
    dense = fql.plan(
        h.to_matrix(levels),
        state,
        [0.0, 0.2],
        dimensions=dimensions,
        collapse_operators=[(loss.to_matrix(levels), 0.3)],
    )
    assert direct.to_dict() == dense.to_dict()
    restored = fql.LindbladPlan.from_json(direct.to_json())
    assert restored.identity == direct.identity
    torch.testing.assert_close(
        fql.run(direct).populations, fql.run(restored).populations, atol=0, rtol=0
    )


@pytest.mark.integration
@pytest.mark.parametrize("levels", [3, 4, 5])
def test_two_excitation_sector_is_closed_above_the_required_cutoff(levels: int) -> None:
    result = fql.run(
        _interference(),
        _pair(levels),
        [0.0, math.pi / (4 * 0.7)],
        dimensions=(levels, levels),
        method="krylov-exponential",
    )
    assert result.populations[-1, 2].item() == pytest.approx(0.5, abs=2e-12)
    assert result.populations[-1, 2 * levels].item() == pytest.approx(0.5, abs=2e-12)
    assert result.populations[-1, levels + 1].item() == pytest.approx(0.0, abs=2e-12)
    # This fixed-excitation sector is exact at three levels; it is not a
    # convergence claim for arbitrary driven oscillator states.


def test_two_levels_cannot_represent_two_photon_bunching() -> None:
    result = fql.run(
        _interference(),
        _pair(2),
        [0.0, math.pi / (4 * 0.7)],
        dimensions=(2, 2),
        method="krylov-exponential",
    )
    assert result.populations[-1, 3].item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_loss_observable_and_spectator_mode_order(precision: str) -> None:
    dtype = getattr(torch, precision)
    state = torch.zeros(6, dtype=dtype)
    state[5] = 1  # |mode0=1, mode1=2>; loss acts on mode 1 only.
    times = torch.linspace(0.0, 0.5, 11, dtype=torch.float64)
    result = evolve_density_matrix(
        boson.BosonOperator(),
        state,
        None,
        times,
        [(boson.annihilate(1), 0.4)],
        {"n0": boson.number(0), "n1": boson.number(1)},
        dimensions=(2, 3),
        dtype=dtype,
        method="krylov-exponential",
    )
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        result.observables["n0"],
        torch.ones(11, dtype=result.times.dtype),
        atol=tolerance,
        rtol=0,
    )
    torch.testing.assert_close(
        result.observables["n1"],
        2 * torch.exp(-0.4 * result.times),
        atol=tolerance,
        rtol=0,
    )
    assert result.populations[:, :3].abs().max().item() < tolerance


@pytest.mark.parametrize(
    "method", ["runge-kutta", "crank-nicolson", "krylov-exponential"]
)
def test_boson_initial_state_batch_gradients_and_adjoint(method: str) -> None:
    weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    ground = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128))
    excited = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128))
    rho = (1 - weight) * ground + weight * excited
    times = torch.linspace(0.0, 0.5, 21, dtype=torch.float64)
    loss = [(boson.annihilate(0), 0.4)]
    h = boson.number(0)
    options = fq.ExecutionOptions(require_gradients=True, batch_size=2)
    planned = fql.plan(
        h,
        torch.stack((rho, ground)),
        times,
        collapse_operators=loss,
        dimensions=(3,),
        method=method,
        options=options,
    )
    result = fql.run(planned)
    (derivative,) = torch.autograd.grad(result.populations[-1, 0, 1], weight)
    assert derivative.item() == pytest.approx(math.exp(-0.2), abs=1e-6)
    cotangent = fql.adjoint_gradient(
        h,
        rho.detach(),
        times,
        lambda trajectory: trajectory[-1, 1, 1].real,
        dimensions=(3,),
        collapse_operators=loss,
        method=method,
    )
    torch.testing.assert_close(
        (cotangent.conj() * (excited - ground)).sum().real,
        derivative,
        atol=2e-12,
        rtol=0,
    )
    with pytest.raises(EvolutionValidationError, match="gradient"):
        fql.run(fql.LindbladPlan.from_json(planned.to_json()))


@pytest.mark.parametrize("field", ["hamiltonian", "collapse_operators", "observables"])
@pytest.mark.parametrize("dimensions", [None, (3,)])
def test_boson_inputs_require_explicit_dimensions_and_in_range_modes(
    field: str, dimensions: tuple[int, ...] | None
) -> None:
    size = 2 if dimensions is None else 3
    operator = boson.number(0 if dimensions is None else 1)
    h = torch.zeros((size, size), dtype=torch.complex128)
    collapse = [(operator, 0.1)] if field == "collapse_operators" else None
    observables = {"number": operator} if field == "observables" else None
    with pytest.raises(EvolutionValidationError) as caught:
        plan_density_matrix_evolution(
            operator if field == "hamiltonian" else h,
            [1.0] + [0.0] * (size - 1),
            1 if dimensions is None else None,
            [0.0, 0.1],
            collapse,
            observables,
            dimensions=dimensions,
        )
    assert caught.value.code == (
        "boson_dimensions_required" if dimensions is None else "invalid_boson_operator"
    )
    assert caught.value.field.startswith(field)


@pytest.mark.parametrize("field", ["hamiltonian", "observables"])
def test_nonhermitian_boson_inputs_are_refused(field: str) -> None:
    with pytest.raises(EvolutionValidationError) as caught:
        plan_density_matrix_evolution(
            boson.annihilate(0) if field == "hamiltonian" else boson.number(0),
            [1.0, 0.0, 0.0],
            None,
            [0.0, 0.1],
            observables={"a": boson.annihilate(0)} if field == "observables" else None,
            dimensions=(3,),
        )
    assert caught.value.code == (
        "non_hermitian_hamiltonian"
        if field == "hamiltonian"
        else "non_hermitian_observable"
    )


def test_existing_boson_materialization_ceiling_remains_enforced() -> None:
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(boson.number(0), "0" * 11, [0.0, 0.1], dimensions=(2,) * 11)
    assert caught.value.code == "invalid_boson_operator"
    assert "ceiling" in str(caught.value)


def test_vacuum_only_space_uses_the_declared_identity_factor() -> None:
    result = fql.run(boson.identity(), [1.0], [0.0, 0.1], dimensions=(1,))
    torch.testing.assert_close(
        result.populations, torch.ones((2, 1), dtype=torch.float64), atol=0, rtol=0
    )
