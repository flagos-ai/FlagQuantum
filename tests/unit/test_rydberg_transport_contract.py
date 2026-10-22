"""Unchanged physics consumers now exercise the native proposal 081 input.

Only the three admitted fixtures change; physical, gradient and replay
assertions remain the independent consumers established before implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.rydberg_native_dynamics import native_pair_plan
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit


class PairPlanner(Protocol):
    def __call__(
        self,
        *,
        intervals: int = 128,
        method: str = "krylov-exponential",
        loss_rate: float = 0.2,
        efficiency: float = 0.8,
        batch_size: int | None = None,
    ) -> fql.LindbladPlan: ...


@pytest.fixture
def pair_planner() -> PairPlanner:
    return native_pair_plan


class PulsePlanner(Protocol):
    def __call__(
        self, amplitude: torch.Tensor, phase: float, method: str
    ) -> fql.LindbladPlan: ...


@pytest.fixture
def pulse_planner() -> PulsePlanner:
    def build(amplitude: torch.Tensor, phase: float, method: str) -> fql.LindbladPlan:
        program = RydbergProgram(
            atoms=(("a", "trap"),),
            sites={"trap": (0.0, 0.0)},
            length_unit="um",
            time_unit="us",
            c6=0.0,
            loss_rate=0.0,
            minimum_separation=1.0,
            drive_channels=("laser",),
        )
        program.pulse(
            "a",
            start=0.0,
            duration=0.5,
            amplitude=fq.Parameter("omega"),
            phase=phase,
            channel="laser",
        )
        return fql.plan(
            program,
            program.basis_state(("g",)),
            torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
            parameter_values=[{"omega": amplitude}] * 200,
            outputs=fq.expectation(program.projector({"a": "r"}), name="rydberg"),
            options=fq.ExecutionOptions(require_gradients=amplitude.requires_grad),
            return_density_matrices=True,
            method=method,
        )

    return build


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("efficiency", [0.0, 0.8, 1.0])
def test_linear_path_loss_detection_axes_and_replay(
    pair_planner: PairPlanner, method: str, batch_size: int | None, efficiency: float
) -> None:
    count = 200
    plan = pair_planner(
        intervals=count, method=method, batch_size=batch_size, efficiency=efficiency
    )
    result = fql.run(plan)
    t = torch.linspace(0.0, 0.5, count + 1, dtype=torch.float64)
    survival = torch.exp(-0.4 * t)
    no_flag = (1 - efficiency * (1 - torch.exp(-0.2 * t))).square()
    values = 64 / (2 + 4 * (t[:-1] + t[1:]) / 2) ** 6
    frozen_phase = torch.cat((torch.zeros(1), torch.cumsum(values * torch.diff(t), 0)))
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    shape = (count + 1,) if batch_size is None else (count + 1, batch_size)
    for name, expected in (
        ("survival", survival),
        ("rr", survival / 2),
        ("coherence", survival * torch.cos(frozen_phase)),
        ("no_flag", no_flag),
    ):
        actual = result.expectation(name)
        assert actual.shape == shape
        active = actual if batch_size is None else actual[:, 0]
        torch.testing.assert_close(active, expected, atol=tolerance, rtol=0)
        if batch_size == 2:
            dark = (1 - efficiency) ** 2 if name == "no_flag" else 0
            torch.testing.assert_close(
                actual[:, 1], torch.full_like(t, dark), atol=tolerance, rtol=0
            )
    population_sum = result.populations.sum(-1)
    torch.testing.assert_close(
        population_sum, torch.ones_like(population_sum), atol=tolerance, rtol=0
    )
    active_rr = result.expectation("rr")
    conditional = active_rr / result.expectation("survival")
    if batch_size is not None:
        conditional = conditional[:, 0]
    torch.testing.assert_close(
        conditional, torch.full_like(t, 0.5), atol=tolerance, rtol=0
    )
    restored = fql.LindbladPlan.from_json(plan.to_json())
    replay = fql.run(restored)
    assert torch.equal(result.populations, replay.populations)
    assert result.density_matrices is not None
    assert replay.density_matrices is not None
    assert torch.equal(result.density_matrices, replay.density_matrices)
    for name in ("survival", "rr", "coherence", "no_flag"):
        assert torch.equal(result.expectation(name), replay.expectation(name))


def test_geometry_convergence_is_separate_from_frozen_solver_accuracy(
    pair_planner: PairPlanner,
) -> None:
    phase = 64 / 20 * (2**-5 - 4**-5)
    errors = []
    for count in (32, 64, 128):
        result = fql.run(pair_planner(intervals=count, loss_rate=0))
        t = torch.linspace(0.0, 0.5, count + 1, dtype=torch.float64)
        values = 64 / (2 + 4 * (t[:-1] + t[1:]) / 2) ** 6
        frozen = torch.sum(values * torch.diff(t)).item()
        actual = result.expectation("coherence")[-1].item()
        assert abs(actual - math.cos(frozen)) < 1e-10
        errors.append(abs(actual - math.cos(phase)))
        torch.testing.assert_close(
            result.expectation("survival"), torch.ones_like(t), atol=1e-10, rtol=0
        )
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


def test_frozen_pair_matches_independent_dense_liouvillian(
    pair_planner: PairPlanner,
) -> None:
    # Independent dense exponentiation is an oracle, not another production solver.
    plan = pair_planner(intervals=2)
    result = fql.run(plan)
    identity = torch.eye(9, dtype=torch.complex128)
    local_identity = torch.eye(3, dtype=torch.complex128)
    rr = torch.zeros((9, 9), dtype=torch.complex128)
    rr[4, 4] = 1
    initial = torch.zeros(9, dtype=torch.complex128)
    initial[0] = initial[4] = 1 / math.sqrt(2)
    density = initial[:, None] * initial.conj()[None, :]
    dissipator = torch.zeros((81, 81), dtype=torch.complex128)
    for level in (0, 1):
        local_jump = torch.zeros((3, 3), dtype=torch.complex128)
        local_jump[2, level] = math.sqrt(0.2)
        for jump in (
            torch.kron(local_jump, local_identity),
            torch.kron(local_identity, local_jump),
        ):
            gram = jump.mH @ jump
            dissipator += torch.kron(jump, jump.conj()) - 0.5 * (
                torch.kron(gram, identity) + torch.kron(identity, gram.T.contiguous())
            )
    assert result.density_matrices is not None
    for index, midpoint in enumerate((0.125, 0.375), 1):
        hamiltonian = (64 / (2 + 4 * midpoint) ** 6) * rr
        generator = (
            -1j
            * (
                torch.kron(hamiltonian, identity)
                - torch.kron(identity, hamiltonian.T.contiguous())
            )
            + dissipator
        )
        density = (torch.matrix_exp(0.25 * generator) @ density.flatten()).reshape(9, 9)
        torch.testing.assert_close(
            result.density_matrices[index], density, atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_pulse_phase_convention_and_control_gradient(
    pulse_planner: PulsePlanner, method: str, phase: float
) -> None:
    amplitude = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    result = fql.run(pulse_planner(amplitude, phase, method))
    (derivative,) = torch.autograd.grad(result.expectation("rydberg")[-1], amplitude)
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert derivative.item() == pytest.approx(0.25 * math.sin(0.35), abs=tolerance)
    vector = torch.tensor(
        [
            math.cos(0.175),
            -1j * complex(math.cos(phase), -math.sin(phase)) * math.sin(0.175),
            0,
        ],
        dtype=torch.complex128,
    )
    assert result.density_matrices is not None
    torch.testing.assert_close(
        result.density_matrices[-1],
        vector[:, None] * vector.conj()[None, :],
        atol=tolerance,
        rtol=0,
    )


def test_live_torch_optimizer_and_finite_difference(
    pulse_planner: PulsePlanner,
) -> None:
    amplitude = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    optimizer = torch.optim.SGD([amplitude], lr=0.2)

    def value(control: torch.Tensor) -> torch.Tensor:
        return fql.run(pulse_planner(control, 0.7, "runge-kutta")).expectation(
            "rydberg"
        )[-1]

    optimizer.zero_grad()
    initial = value(amplitude)
    (derivative,) = torch.autograd.grad(initial, amplitude, retain_graph=True)
    epsilon = 1e-5
    difference = (
        value(amplitude.detach() + epsilon) - value(amplitude.detach() - epsilon)
    ) / (2 * epsilon)
    torch.testing.assert_close(derivative, difference, atol=1e-9, rtol=0)
    loss = (initial - 0.1).square()
    torch.autograd.backward(loss)
    optimizer.step()
    assert (value(amplitude) - 0.1).square().item() < loss.item()


def test_json_replay_does_not_claim_to_restore_a_control_graph(
    pulse_planner: PulsePlanner,
) -> None:
    amplitude = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    plan = pulse_planner(amplitude, 0.7, "runge-kutta")
    restored = fql.LindbladPlan.from_json(plan.to_json())
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(restored)
    assert caught.value.code == "gradients_not_available"


@pytest.fixture
def initial_adjoint() -> Callable[..., torch.Tensor]:
    def native_adjoint(
        hamiltonian: torch.Tensor,
        initial: torch.Tensor,
        times: torch.Tensor,
        cost: Callable[[torch.Tensor], torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        # This fixture replaces one fixed oracle drive, not a matrix-to-program compiler.
        expected = torch.zeros((3, 3), dtype=torch.complex128)
        expected[0, 1] = expected[1, 0] = 0.35
        assert torch.equal(hamiltonian, expected)
        program = RydbergProgram(
            atoms=(("a", "trap"),),
            sites={"trap": (0.0, 0.0)},
            length_unit="um",
            time_unit="us",
            c6=0.0,
            loss_rate=0.0,
            minimum_separation=1.0,
            drive_channels=("laser",),
        )
        program.pulse("a", start=0.0, duration=0.5, amplitude=0.7, channel="laser")
        return fql.adjoint_gradient(program, initial, times, cost, **kwargs)

    return native_adjoint


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_initial_adjoint_matches_independent_discrete_map(
    initial_adjoint: Callable[..., torch.Tensor], method: str
) -> None:
    hamiltonian = torch.zeros((3, 3), dtype=torch.complex128)
    hamiltonian[0, 1] = hamiltonian[1, 0] = 0.35
    observable = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128))
    initial = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128))
    identity = torch.eye(3, dtype=torch.complex128)
    generator = -1j * (
        torch.kron(hamiltonian, identity)
        - torch.kron(identity, hamiltonian.T.contiguous())
    )
    step = 0.5 / 200
    eye = torch.eye(9, dtype=torch.complex128)
    if method == "crank-nicolson":
        half = step / 2
        substep = torch.linalg.solve(
            eye - half * generator / 2, eye + half * generator / 2
        )
        propagator = torch.linalg.matrix_power(substep, 400)
    elif method == "runge-kutta":
        scaled = (step / 2) * generator
        substep = eye + scaled + scaled @ scaled / 2
        substep += torch.linalg.matrix_power(scaled, 3) / 6
        substep += torch.linalg.matrix_power(scaled, 4) / 24
        propagator = torch.linalg.matrix_power(substep, 400)
    else:
        propagator = torch.matrix_exp(0.5 * generator)
    expected = (propagator.mH @ observable.flatten()).reshape(3, 3)
    actual = initial_adjoint(
        hamiltonian,
        initial,
        torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
        lambda trajectory: torch.real(torch.trace(observable @ trajectory[-1])),
        dimensions=(3,),
        method=method,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
