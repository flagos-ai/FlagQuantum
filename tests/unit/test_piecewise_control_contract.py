"""Proposal 073 consumer assertions, initially backed by static composition.

The test-only fake uses today's public API. The scheduled Simulation boundary
must replace it without changing these physical assertions. It is not a second
production integrator or a claim that the proposed API is available.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.parameters import bind_parameter_value
from flagquantum.simulation.lindblad_integrators import INTEGRATOR_NAMES

pytestmark = pytest.mark.unit


def _static_composition_fake(
    hamiltonian, initial_state, times, *, parameter_values, **kwargs
):
    states, populations = [], []
    state = initial_state
    for k, values in enumerate(parameter_values):
        result = fql.run(
            bind_parameter_value(hamiltonian, values),
            state,
            times[k : k + 2],
            return_density_matrices=True,
            **kwargs,
        )
        if k == 0:
            states.append(result.density_matrices[0])
            populations.append(result.populations[0])
        state = result.density_matrices[-1]
        states.append(state)
        populations.append(result.populations[-1])
    return replace(
        result,
        density_matrices=torch.stack(states),
        populations=torch.stack(populations),
    )


@pytest.fixture(params=[_static_composition_fake], ids=["static-composition-fake"])
def run_evolution(request):
    return request.param


def _transition(first, second):
    matrix = torch.zeros((3, 3), dtype=torch.complex128)
    matrix[first, second] = matrix[second, first] = 1
    return matrix


@pytest.mark.parametrize("method", INTEGRATOR_NAMES)
def test_nonuniform_qutrit_population_and_live_gradients(run_evolution, method):
    drive = fq.Parameter("drive")
    omega = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)
    times = [2.0, 2.1, 2.3, 2.35]
    result = run_evolution(
        [{"operator": _transition(1, 2), "coefficient": 0.5 * drive}],
        [0.0, 1.0, 0.0],
        times,
        dimensions=(3,),
        parameter_values=[{drive: value} for value in omega.unbind()],
        method=method,
        options=fq.ExecutionOptions(require_gradients=True),
    )
    dt = omega.new_tensor([0.1, 0.2, 0.05])
    angle = (omega.detach() * dt).sum()
    expected = torch.sin(angle / 2) ** 2
    expected_gradient = 0.5 * dt * torch.sin(angle)
    (gradient,) = torch.autograd.grad(result.populations[-1, 2], omega)
    tolerance = {
        "runge-kutta": 3e-5,
        "crank-nicolson": 0.004,
        "krylov-exponential": 1e-10,
    }[method]
    torch.testing.assert_close(
        result.populations[-1, 2], expected, atol=tolerance, rtol=0
    )
    torch.testing.assert_close(gradient, expected_gradient, atol=tolerance, rtol=0)
    assert result.populations.shape == (4, 3)
    torch.testing.assert_close(result.populations[0], omega.new_tensor([0.0, 1.0, 0.0]))


def test_ordered_dissipative_pulses_match_independent_superoperator(run_evolution):
    first, second = fq.Parameter("first"), fq.Parameter("second")
    x01, x12 = _transition(0, 1), _transition(1, 2)
    template = [
        {"operator": x01, "coefficient": first},
        {"operator": x12, "coefficient": second},
    ]
    bindings = [
        {"first": math.pi / 0.2, "second": 0.0},
        {"first": 0.0, "second": math.pi / 0.4},
    ]
    decay = torch.zeros_like(x01)
    decay[1, 2] = 1
    result = run_evolution(
        template,
        [1.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        parameter_values=bindings,
        collapse_operators=[(decay, 0.2)],
        method="krylov-exponential",
    )
    eye = torch.eye(3, dtype=torch.complex128)
    a = decay.mH @ decay
    dissipator = 0.2 * (
        torch.kron(decay, decay.conj())
        - 0.5 * (torch.kron(a, eye) + torch.kron(eye, a.T.contiguous()))
    )
    reference = torch.zeros((3, 3), dtype=torch.complex128)
    reference[0, 0] = 1
    for h, dt in [(math.pi / 0.2 * x01, 0.1), (math.pi / 0.4 * x12, 0.2)]:
        generator = -1j * (torch.kron(h, eye) - torch.kron(eye, h.T.contiguous()))
        reference = (
            torch.matrix_exp(dt * (generator + dissipator)) @ reference.flatten()
        ).reshape(3, 3)
    torch.testing.assert_close(
        result.density_matrices[-1], reference, atol=1e-10, rtol=0
    )
    reversed_result = run_evolution(
        template,
        [1.0, 0.0, 0.0],
        [0.0, 0.2, 0.3],
        dimensions=(3,),
        parameter_values=list(reversed(bindings)),
        method="krylov-exponential",
    )
    assert result.populations[-1, 2].item() > 0.9
    assert reversed_result.populations[-1, 2].item() < 1e-12


def test_batch_and_shared_control_accumulate_one_gradient(run_evolution):
    drive = fq.Parameter("drive")
    omega = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    result = run_evolution(
        [{"operator": _transition(1, 2), "coefficient": 0.5 * drive}],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        parameter_values=[{"drive": omega}, {"drive": omega}],
        method="krylov-exponential",
        options=fq.ExecutionOptions(require_gradients=True),
    )
    assert result.populations.shape == (3, 2, 3)
    (gradient,) = torch.autograd.grad(result.populations[-1, 0, 2], omega)
    assert gradient.item() == pytest.approx(0.15 * math.sin(0.3), abs=1e-10)
