"""Independent scalar physics for the approved fixed-rate qutrit cascade."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.integration


def reference_controls() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0], [0.6, 0.3], [-0.2, 0.7], [0.8, -0.1]], dtype=torch.float64
    )


def reference_inputs() -> tuple[torch.Tensor, ...]:
    basis = torch.eye(3, dtype=torch.complex128)
    vector = torch.tensor([1, 1j, 1], dtype=torch.complex128) / math.sqrt(3)
    coherent = 0.7 * basis / 3 + 0.3 * torch.outer(vector, vector.conj())
    return (*(torch.outer(row, row.conj()) for row in basis), coherent)


def reference_generator(
    x: float,
    y: float,
    rates: tuple[float, float] = (0.4, 0.7),
    *,
    undriven: bool = False,
) -> torch.Tensor:
    """Apply the scalar equation to E_ab columns, independently of package code."""
    drive = complex(x, -y) / 2
    h = torch.tensor(
        [
            [0, drive, 0],
            [drive.conjugate(), 0.15, math.sqrt(2) * drive],
            [0, math.sqrt(2) * drive.conjugate(), -2.7],
        ],
        dtype=torch.complex128,
    )
    if undriven:
        h = torch.zeros_like(h)
    columns = []
    for a in range(3):
        for b in range(3):
            e = torch.zeros((3, 3), dtype=torch.complex128)
            e[a, b] = 1
            derivative = -1j * (h @ e - e @ h)
            for lower, upper, rate in ((0, 1, rates[0]), (1, 2, rates[1])):
                jump = torch.zeros_like(e)
                jump[lower, upper] = 1
                occupied = jump.mH @ jump
                derivative += rate * (
                    jump @ e @ jump.mH - (occupied @ e + e @ occupied) / 2
                )
            columns.append(derivative.reshape(9))
    return torch.stack(columns, dim=1)


def reference_states(
    controls: torch.Tensor,
    initial: torch.Tensor,
    rates: tuple[float, float] = (0.4, 0.7),
    *,
    undriven: bool = False,
) -> torch.Tensor:
    state = initial.clone()
    rows = [state]
    for x, y in controls.detach().tolist():
        propagator = torch.matrix_exp(
            0.25 * reference_generator(x, y, rates, undriven=undriven)
        )
        state = (propagator @ state.reshape(9)).reshape(3, 3)
        rows.append(state)
    return torch.stack(rows)


@pytest.mark.parametrize("rates", [(0.4, 0.7), (0.0, 0.7), (0.4, 0.0), (0.0, 0.0)])
def test_independent_reference_against_analytic_cascade_and_coherences(
    rates: tuple[float, float],
) -> None:
    g10, g21 = rates
    controls = torch.zeros((4, 2), dtype=torch.float64)
    inputs = reference_inputs()
    level2 = reference_states(controls, inputs[2], rates, undriven=True)
    level1 = reference_states(controls, inputs[1], rates, undriven=True)
    coherent = reference_states(controls, inputs[3], rates, undriven=True)
    for index in range(5):
        time = index / 4
        p2 = math.exp(-g21 * time)
        p1 = (
            g21 * (p2 - math.exp(-g10 * time)) / (g10 - g21)
            if g10 != g21
            else g21 * time * p2
        )
        torch.testing.assert_close(
            level2[index].diagonal().real,
            torch.tensor([1 - p1 - p2, p1, p2], dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )
        assert abs(float(level1[index, 1, 1].real) - math.exp(-g10 * time)) <= 1e-10
        for a, b, decay in ((0, 1, g10), (0, 2, g21), (1, 2, g10 + g21)):
            expected = inputs[3][a, b] * math.exp(-decay * time / 2)
            torch.testing.assert_close(
                coherent[index, a, b], expected, atol=1e-10, rtol=0
            )


@pytest.mark.parametrize("initial", reference_inputs())
def test_driven_density_order_and_physical_invariants(initial: torch.Tensor) -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = reference_controls()
    result = fql.run(relaxation_plan(controls, initial=initial))
    assert result.density_matrices is not None
    rows = result.density_matrices
    torch.testing.assert_close(
        rows, reference_states(controls, initial), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        rows.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.ones(5, dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )
    assert float(abs(rows - rows.mH).max()) <= 1e-10
    assert float(torch.linalg.eigvalsh(rows).min()) >= -1e-10
    reverse = fql.run(relaxation_plan(controls.flip(0), initial=initial))
    assert reverse.density_matrices is not None
    torch.testing.assert_close(
        reverse.density_matrices,
        reference_states(controls.flip(0), initial),
        atol=1e-10,
        rtol=0,
    )
    assert float(abs(reverse.density_matrices[-1] - rows[-1]).max()) > 1e-3
    assert float(result.expectation("leakage").max()) > 0


@pytest.mark.parametrize("rates", [(0.4, 0.7), (0.0, 0.7), (0.4, 0.0), (0.0, 0.0)])
def test_production_undriven_cascade_and_coherence(rates: tuple[float, float]) -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = torch.zeros((4, 2), dtype=torch.float64)
    for initial in reference_inputs():
        result = fql.run(
            relaxation_plan(controls, initial=initial, rates=rates, undriven=True)
        )
        assert result.density_matrices is not None
        torch.testing.assert_close(
            result.density_matrices,
            reference_states(controls, initial, rates, undriven=True),
            atol=1e-10,
            rtol=0,
        )


@pytest.mark.parametrize("level", [1, 2])
def test_all_eight_control_gradients(level: int) -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = reference_controls().requires_grad_()
    result = fql.run(relaxation_plan(controls))
    name = "excited" if level == 1 else "leakage"
    (gradient,) = torch.autograd.grad(result.expectation(name)[-1], controls)
    step = 1e-4
    for pulse in range(4):
        for quadrature in range(2):
            values = []
            for multiple in (-2, -1, 1, 2):
                shifted = controls.detach().clone()
                shifted[pulse, quadrature] += multiple * step
                values.append(
                    reference_states(shifted, reference_inputs()[0])[
                        -1, level, level
                    ].real
                )
            expected = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (
                12 * step
            )
            assert abs(float(gradient[pulse, quadrature] - expected)) <= 1e-9


def test_runge_kutta_refinement() -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = reference_controls()
    reference = reference_states(controls, reference_inputs()[0])
    errors = []
    for subdivisions in (1, 2, 4, 8):
        result = fql.run(
            relaxation_plan(controls, subdivisions=subdivisions, method="runge-kutta")
        )
        assert result.density_matrices is not None
        errors.append(
            float(abs(result.density_matrices[::subdivisions] - reference).max())
        )
    assert all(a / b > 10 for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-6


def test_training_detached_replay_and_independent_final_populations() -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = torch.nn.Parameter(reference_controls())
    optimizer = torch.optim.Adam([controls], lr=0.08)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        result = fql.run(relaxation_plan(controls))
        loss = (
            1 - result.expectation("excited")[-1]
        ).square() + 2 * result.expectation("leakage")[-1]
        torch.autograd.backward(loss)
        assert controls.grad is not None and bool(torch.isfinite(controls.grad).all())
        losses.append(float(loss.detach()))
        optimizer.step()
    assert all(math.isfinite(value) for value in losses)
    plan = relaxation_plan(controls.detach())
    result = fql.run(plan)
    final_loss = float(
        (1 - result.expectation("excited")[-1]).square()
        + 2 * result.expectation("leakage")[-1]
    )
    assert final_loss <= 0.9 * losses[0]
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert result.density_matrices is not None and replay.density_matrices is not None
    assert torch.equal(result.density_matrices, replay.density_matrices)
    for name in ("ground", "excited", "leakage"):
        assert torch.equal(result.expectation(name), replay.expectation(name))
    torch.testing.assert_close(
        result.density_matrices,
        reference_states(controls.detach(), reference_inputs()[0]),
        atol=1e-10,
        rtol=0,
    )


def test_zero_rate_limit_retains_closed_model() -> None:
    import flagquantum.lindblad as fql
    from examples.driven_three_level import (
        pulse_plan,
    )
    from examples.driven_three_level import (
        reference_states as closed_reference,
    )
    from examples.driven_three_level_relaxation import relaxation_plan

    controls = reference_controls()
    closed = fql.run(pulse_plan(controls))
    result = fql.run(relaxation_plan(controls, rates=(0.0, 0.0)))
    assert result.density_matrices is not None and closed.density_matrices is not None
    torch.testing.assert_close(
        result.density_matrices, closed.density_matrices, atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        result.density_matrices, closed_reference(controls), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize(
    "rates", [(-0.4, 0.7), (float("nan"), 0.7), (0.4, float("inf"))]
)
def test_existing_rate_admission(rates: tuple[float, float]) -> None:
    from examples.driven_three_level_relaxation import relaxation_plan

    with pytest.raises(ValueError):
        relaxation_plan(reference_controls(), rates=rates)
