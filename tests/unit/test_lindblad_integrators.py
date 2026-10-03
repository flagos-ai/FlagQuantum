from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.lindblad import LindbladPlan
from flagquantum.lindblad import _plan as plan_module
from flagquantum.simulation import (
    EvolutionValidationError,
    amplitude_damping,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)
from flagquantum.simulation import lindblad as lindblad_module
from flagquantum.simulation import lindblad_integrators as integrators_module
from flagquantum.simulation.matrix_free_linear_solve import gmres

pytestmark = pytest.mark.unit

EXCITED = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
ZERO_HAMILTONIAN = torch.zeros((2, 2), dtype=torch.complex128)


def _exact_amplitude_damping(rate: float, times: torch.Tensor) -> torch.Tensor:
    """Return the closed-form density matrix of amplitude damping from ``|1>``."""

    return torch.stack(
        [
            torch.tensor(
                [
                    [1.0 - math.exp(-rate * float(time)), 0.0],
                    [0.0, math.exp(-rate * float(time))],
                ],
                dtype=torch.complex128,
            )
            for time in times
        ]
    )


def _decay_error(rate: float, steps: int, method: str) -> float:
    """Return the worst deviation from the closed-form decay on a uniform grid."""

    times = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    result = evolve_density_matrix(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        [amplitude_damping(rate=rate, wire=0)],
        return_density_matrices=True,
        method=method,
    )
    assert result.density_matrices is not None
    deviation = (result.density_matrices - _exact_amplitude_damping(rate, times)).abs()
    return float(deviation.max())


def _dense_system(size: int, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((size, size), generator=generator, dtype=torch.float64)
    imaginary = torch.randn((size, size), generator=generator, dtype=torch.float64)
    matrix = (real + 1j * imaginary).to(torch.complex128)
    matrix = matrix + size * torch.eye(size, dtype=torch.complex128)
    vector = torch.randn((size,), generator=generator, dtype=torch.float64)
    return matrix, vector.to(torch.complex128)


def test_gmres_solves_a_complex_system_and_reports_the_measured_residual() -> None:
    matrix, rhs = _dense_system(6)
    solution, residual = gmres(
        lambda vector: matrix @ vector, rhs, tolerance=1e-12, max_iterations=200
    )

    expected = torch.linalg.solve(matrix, rhs)
    relative = float(torch.linalg.vector_norm(rhs - matrix @ solution)) / float(
        torch.linalg.vector_norm(rhs)
    )
    assert torch.allclose(solution, expected, atol=1e-10, rtol=0.0)
    # The returned residual is remeasured by applying the operator, and the
    # independent recomputation agrees to 1e-14. Reporting an internal Krylov
    # estimate instead diverges from this value by up to 1e15 for a Lindblad
    # step, so this assertion is the honesty guard on the returned number.
    assert residual == pytest.approx(relative, abs=1e-14)
    assert residual <= 1e-12


def test_gmres_on_a_zero_right_hand_side_returns_zero() -> None:
    matrix, _ = _dense_system(4)
    rhs = torch.zeros(4, dtype=torch.complex128)

    solution, residual = gmres(
        lambda vector: matrix @ vector, rhs, tolerance=1e-12, max_iterations=10
    )

    assert torch.equal(solution, rhs)
    assert residual == 0.0


def test_gmres_refuses_a_non_flat_right_hand_side() -> None:
    matrix, _ = _dense_system(4)
    rhs = torch.zeros((2, 2), dtype=torch.complex128)

    with pytest.raises(ValueError, match="flat vector"):
        gmres(lambda vector: matrix @ vector, rhs, tolerance=1e-12, max_iterations=10)


def test_gmres_reports_the_residual_it_actually_achieved_when_it_stops_early() -> None:
    matrix, rhs = _dense_system(6)

    solution, residual = gmres(
        lambda vector: matrix @ vector, rhs, tolerance=1e-14, max_iterations=1
    )

    relative = float(torch.linalg.vector_norm(rhs - matrix @ solution)) / float(
        torch.linalg.vector_norm(rhs)
    )
    assert residual > 1e-14
    assert residual == pytest.approx(relative, abs=1e-14)


def test_the_declared_integrator_vocabulary_is_closed() -> None:
    assert integrators_module.INTEGRATOR_NAMES == ("runge-kutta", "crank-nicolson")
    assert integrators_module.DEFAULT_INTEGRATOR == "runge-kutta"
    assert integrators_module.integrator_order("runge-kutta") == 4
    assert integrators_module.integrator_order("crank-nicolson") == 2
    assert integrators_module.SUBSTEPS_PER_INTERVAL == 2


def test_runge_kutta_converges_at_fourth_order() -> None:
    coarse = _decay_error(1.0, 8, "runge-kutta")
    fine = _decay_error(1.0, 64, "runge-kutta")

    order = math.log2(coarse / fine) / 3
    # Measured 4.928e-08 at h=1/8 and 1.150e-11 at h=1/64 gives order 4.02.
    # A second-order explicit scheme yields order 2.0 here, and a first-order
    # scheme yields 1.0, so the bracket separates the declared order.
    assert 3.8 < order < 4.3


def test_crank_nicolson_converges_at_second_order() -> None:
    coarse = _decay_error(1.0, 8, "crank-nicolson")
    fine = _decay_error(1.0, 64, "crank-nicolson")

    order = math.log2(coarse / fine) / 3
    # Measured 1.198e-04 at h=1/8 and 1.871e-06 at h=1/64 gives order 2.00.
    # Backward Euler would give order 1.0, and reusing the explicit weights
    # would give order 4.0.
    assert 1.85 < order < 2.15


def test_crank_nicolson_stays_bounded_where_runge_kutta_diverges() -> None:
    times = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
    operators = [amplitude_damping(rate=1000.0, wire=0)]

    explicit = evolve_density_matrix(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        operators,
        return_density_matrices=True,
        method="runge-kutta",
    )
    implicit = evolve_density_matrix(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        operators,
        return_density_matrices=True,
        method="crank-nicolson",
    )
    assert explicit.density_matrices is not None
    assert implicit.density_matrices is not None

    # rate * h is 50 here, far beyond the explicit stability limit. Measured:
    # the explicit trajectory reaches 6.250e+165 with a trace drift of 6.097e+141
    # and reports population_bounded False, while the trapezoidal trajectory
    # stays at unit scale with a trace drift of 1.188e-14. An implicit scheme
    # that accidentally reused the explicit update would diverge identically.
    assert float(explicit.density_matrices.abs().max()) > 1e100
    assert explicit.maximum_trace_drift > 1e100
    assert not explicit.population_bounded

    assert float(implicit.density_matrices.abs().max()) == pytest.approx(1.0, abs=1e-9)
    assert implicit.maximum_trace_drift < 1e-9
    assert implicit.population_bounded
    # The trapezoidal rule is only second-order, so a coarse stiff grid still
    # carries an O(h**2) error of 7.257e-01 against the closed form.
    assert _decay_error(1000.0, 20, "crank-nicolson") == pytest.approx(
        7.256516e-01, abs=1e-6
    )


def test_crank_nicolson_refines_toward_the_exact_stiff_solution() -> None:
    coarse = _decay_error(1000.0, 100, "crank-nicolson")
    fine = _decay_error(1000.0, 1600, "crank-nicolson")

    # Measured 1.836e-01 at h=1/100 and 2.943e-03 at h=1/1600. Explicit
    # Runge-Kutta reaches 2.494e+227 and 3.697e-05 at the same grids, so this
    # assertion is about the implicit scheme remaining finite and convergent,
    # not about it beating the explicit scheme on accuracy where both are stable.
    assert fine < coarse
    assert fine < 5e-3


def test_the_default_integrator_remains_the_explicit_scheme() -> None:
    times = torch.linspace(0.0, 4.0, 41, dtype=torch.float64)
    operators = [amplitude_damping(rate=0.4, wire=0)]

    default = evolve_density_matrix(
        ZERO_HAMILTONIAN, EXCITED, 1, times, operators, return_density_matrices=True
    )
    explicit = evolve_density_matrix(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        operators,
        return_density_matrices=True,
        method="runge-kutta",
    )

    assert default.method == "runge-kutta"
    assert default.order == 4
    assert default.density_matrices is not None
    assert explicit.density_matrices is not None
    assert torch.equal(default.density_matrices, explicit.density_matrices)


def test_the_reported_method_is_the_method_that_ran() -> None:
    times = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    operators = [amplitude_damping(rate=0.4, wire=0)]

    result = evolve_density_matrix(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        operators,
        method="crank-nicolson",
    )

    assert result.method == "crank-nicolson"
    assert result.order == 2
    numerics = result.to_dict()["numerics"]
    assert numerics["method"] == "crank-nicolson"
    assert numerics["order"] == 2


@pytest.mark.parametrize("method", ["magnus", "RK4", "", "runge_kutta", "cranknicolson"])
def test_an_unknown_integration_method_is_refused(method: str) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    with pytest.raises(EvolutionValidationError) as error:
        evolve_density_matrix(ZERO_HAMILTONIAN, EXCITED, 1, times, None, method=method)

    assert error.value.code == "unsupported_method"
    assert error.value.field == "method"
    assert "crank-nicolson" in str(error.value)


@pytest.mark.parametrize("tolerance", [0.0, 1.0, -1e-3, 2.0])
def test_a_solve_tolerance_outside_the_unit_interval_is_refused(
    tolerance: float,
) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    with pytest.raises(EvolutionValidationError) as error:
        evolve_density_matrix(
            ZERO_HAMILTONIAN,
            EXCITED,
            1,
            times,
            None,
            method="crank-nicolson",
            solve_tolerance=tolerance,
        )

    assert error.value.code == "invalid_solve_tolerance"
    assert error.value.field == "solve_tolerance"


@pytest.mark.parametrize("tolerance", ["1e-9", True, None, [1e-9]])
def test_a_non_numeric_solve_tolerance_is_refused(tolerance: object) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    if tolerance is None:
        # None is the documented default and must select the precision default.
        plan = plan_density_matrix_evolution(
            ZERO_HAMILTONIAN,
            EXCITED,
            1,
            times,
            None,
            method="crank-nicolson",
            solve_tolerance=None,
        )
        assert plan.solve_tolerance == 1e-12
        return

    with pytest.raises(EvolutionValidationError) as error:
        plan_density_matrix_evolution(
            ZERO_HAMILTONIAN,
            EXCITED,
            1,
            times,
            None,
            method="crank-nicolson",
            solve_tolerance=tolerance,  # type: ignore[arg-type]
        )

    assert error.value.code == "invalid_solve_tolerance"


@pytest.mark.parametrize("tolerance", [1e-6, 1e-12])
def test_a_solve_tolerance_on_an_explicit_scheme_is_refused(tolerance: float) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    with pytest.raises(EvolutionValidationError) as error:
        plan_density_matrix_evolution(
            ZERO_HAMILTONIAN,
            EXCITED,
            1,
            times,
            None,
            method="runge-kutta",
            solve_tolerance=tolerance,
        )

    assert error.value.code == "invalid_solve_tolerance"
    assert "implicit" in str(error.value)


def test_the_default_solve_tolerance_follows_the_working_precision() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    double = plan_density_matrix_evolution(
        ZERO_HAMILTONIAN, EXCITED, 1, times, None, method="crank-nicolson"
    )
    single = plan_density_matrix_evolution(
        ZERO_HAMILTONIAN,
        EXCITED,
        1,
        times,
        None,
        dtype=torch.complex64,
        method="crank-nicolson",
    )

    assert double.solve_tolerance == 1e-12
    assert single.solve_tolerance == 1e-5


def test_an_explicit_plan_records_no_solve_tolerance() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    plan = plan_density_matrix_evolution(
        ZERO_HAMILTONIAN, EXCITED, 1, times, None, method="runge-kutta"
    )

    assert plan.solve_tolerance is None
    assert plan.to_dict()["numerics"]["solve_tolerance"] is None


def test_a_tighter_solve_tolerance_produces_a_tighter_trajectory() -> None:
    n_wires = 3
    hamiltonian = 0.0 * fq.Z(0)
    for wire in range(n_wires - 1):
        hamiltonian = hamiltonian + 0.5 * fq.Z(wire) @ fq.Z(wire + 1)
    for wire in range(n_wires):
        hamiltonian = hamiltonian + 0.25 * fq.X(wire)
    times = torch.linspace(0.0, 2.0, 41, dtype=torch.float64)
    state = torch.zeros(2**n_wires, dtype=torch.complex128)
    state[0] = 1.0
    initial = torch.outer(state, state.conj())

    def deviation(tolerance: float) -> float:
        result = evolve_density_matrix(
            hamiltonian,
            initial,
            n_wires,
            times,
            return_density_matrices=True,
            method="crank-nicolson",
            solve_tolerance=tolerance,
        )
        assert result.density_matrices is not None
        return float(
            (result.density_matrices - reference.density_matrices).abs().max()
        )

    reference = evolve_density_matrix(
        hamiltonian,
        initial,
        n_wires,
        times,
        return_density_matrices=True,
        method="crank-nicolson",
        solve_tolerance=1e-15,
    )
    assert reference.density_matrices is not None

    loose = deviation(1e-6)
    tight = deviation(1e-12)
    # Measured 1.579e-06 at 1e-6 and 2.245e-12 at 1e-12 against the 1e-15
    # reference, mirroring the reachable residual at each setting.
    assert loose == pytest.approx(1.579142e-06, rel=0.1)
    assert tight == pytest.approx(2.244507e-12, rel=0.1)
    assert tight < loose / 1000


def test_an_unreachable_solve_tolerance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    monkeypatch.setattr(integrators_module, "_SOLVE_RESTART", 1)
    monkeypatch.setattr(lindblad_module, "_solve_max_iterations", lambda dimension: 1)

    with pytest.raises(EvolutionValidationError) as error:
        evolve_density_matrix(
            ZERO_HAMILTONIAN,
            EXCITED,
            1,
            times,
            [amplitude_damping(rate=0.4, wire=0)],
            method="crank-nicolson",
            solve_tolerance=1e-15,
        )

    assert error.value.code == "integration_not_converged"
    assert error.value.field == "solve_tolerance"


def test_the_step_size_is_the_substep_of_every_scheme() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    for method in integrators_module.INTEGRATOR_NAMES:
        plan = plan_density_matrix_evolution(
            ZERO_HAMILTONIAN, EXCITED, 1, times, None, method=method
        )
        assert plan.step_size == pytest.approx(0.125)


def test_the_plan_round_trips_the_integration_method() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128),
        EXCITED,
        times,
        collapse_operators=[amplitude_damping(rate=1.0, wire=0)],
        method="crank-nicolson",
        solve_tolerance=1e-9,
    )

    assert isinstance(planned, LindbladPlan)
    assert planned.method == "crank-nicolson"
    assert planned.order == 2
    assert planned.solve_tolerance == 1e-9
    numerics = planned.to_dict()["decision"]["numerics"]
    assert numerics["method"] == "crank-nicolson"
    assert numerics["order"] == 2
    assert numerics["solve_tolerance"] == 1e-9

    reloaded = LindbladPlan.from_json(planned.to_json())
    assert reloaded.identity == planned.identity
    assert reloaded.method == "crank-nicolson"
    assert reloaded.solve_tolerance == 1e-9
    assert reloaded._execution_request()["method"] == "crank-nicolson"
    assert reloaded._execution_request()["solve_tolerance"] == 1e-9


def test_the_plan_identity_separates_the_integration_methods() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    explicit = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128), EXCITED, times, method="runge-kutta"
    )
    implicit = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128), EXCITED, times, method="crank-nicolson"
    )

    assert explicit.identity != implicit.identity
    assert explicit.solve_tolerance is None
    assert implicit.solve_tolerance == 1e-12


def test_running_a_plan_reports_the_planned_integration_method() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128),
        EXCITED,
        times,
        collapse_operators=[amplitude_damping(rate=1.0, wire=0)],
        method="crank-nicolson",
    )

    result = fq.lindblad.run(planned)

    assert result.method == "crank-nicolson"
    assert result.order == 2
    assert result.plan is not None
    assert result.plan.identity == planned.identity


@pytest.mark.parametrize(
    "override",
    [
        {"method": "crank-nicolson"},
        {"solve_tolerance": 1e-9},
        {"return_density_matrices": True},
    ],
)
def test_a_sealed_plan_is_closed_to_semantic_overrides(
    override: dict[str, object],
) -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128), EXCITED, times, method="crank-nicolson"
    )

    with pytest.raises(TypeError, match="closed to"):
        fq.lindblad.run(planned, **override)


def test_a_tampered_integration_method_is_refused_on_decode() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128), EXCITED, times, method="crank-nicolson"
    )
    payload = planned.to_dict()
    decision = dict(payload["decision"])
    decision["numerics"] = {**decision["numerics"], "method": "magnus"}
    forged = {**payload, "decision": decision}
    # Re-sealing the forged payload clears the identity guard, so the refusal
    # below proves that decoding re-runs the planner with the recorded method
    # rather than trusting the serialized decision.
    forged["identity"] = plan_module._identity(forged)

    with pytest.raises(EvolutionValidationError) as error:
        LindbladPlan.from_dict(forged)

    assert error.value.code == "unsupported_method"


def test_a_tampered_integration_order_is_refused_on_decode() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128), EXCITED, times, method="crank-nicolson"
    )
    payload = planned.to_dict()
    decision = dict(payload["decision"])
    decision["numerics"] = {**decision["numerics"], "order": 4}
    forged = {**payload, "decision": decision}
    forged["identity"] = plan_module._identity(forged)

    with pytest.raises(Exception) as error:
        LindbladPlan.from_dict(forged)

    assert type(error.value).__name__ == "SerializationError"
    assert "does not match its request" in str(error.value)


def test_a_tampered_solve_tolerance_is_refused_on_decode() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    planned = fq.lindblad.plan(
        torch.eye(2, dtype=torch.complex128),
        EXCITED,
        times,
        collapse_operators=[amplitude_damping(rate=1.0, wire=0)],
        method="crank-nicolson",
    )
    payload = planned.to_dict()
    decision = dict(payload["decision"])
    decision["numerics"] = {**decision["numerics"], "solve_tolerance": 1e-3}

    with pytest.raises(Exception) as error:
        LindbladPlan.from_dict({**payload, "decision": decision})

    assert type(error.value).__name__ == "SerializationError"


def test_the_plan_documents_the_integration_cost_not_the_superoperator_size() -> None:
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    for method in integrators_module.INTEGRATOR_NAMES:
        plan = plan_density_matrix_evolution(
            ZERO_HAMILTONIAN, EXCITED, 1, times, None, method=method
        )
        # A matrix-free implicit step never materializes a (4**n, 4**n)
        # superoperator, so the reported footprint stays the 4**n trajectory.
        assert plan.trajectory_bytes == 5 * 4 * 16
        assert plan.dimension == 2
