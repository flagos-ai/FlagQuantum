"""Time-integration schemes for the Lindblad generator.

The generator is applied, never formed: ``derivative`` maps a density matrix to
``d rho / dt``. An explicit scheme is therefore a fixed combination of
applications, an implicit scheme additionally needs a matrix-free linear solve,
and an exponential scheme needs the matrix-free action of the generator's
exponential. No scheme pays for the ``16**n`` entries of the vectorized
generator.

The schemes are not interchangeable in what their accuracy depends on. Runge--
Kutta and Crank--Nicolson are order-``p`` in the step: their error falls as a
power of the step size and vanishes only as the step does. The exponential of a
time-independent generator has no step-size error at all -- the grid is exact up
to the Krylov defect the exponential action measures -- so its accuracy is set
by ``solve_tolerance`` rather than by an order, and it reports no order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from .matrix_free_exponential import exponential_action
from .matrix_free_linear_solve import gmres

RUNGE_KUTTA = "runge-kutta"
CRANK_NICOLSON = "crank-nicolson"
KRYLOV_EXPONENTIAL = "krylov-exponential"

DEFAULT_INTEGRATOR = RUNGE_KUTTA

# The explicit output grid defines the integration step, and every scheme that
# carries a step-size error takes the same number of substeps inside one output
# interval. Schemes therefore differ only by the scheme, and a reported step
# size means the same thing for all of them: the step the scheme actually took.
SUBSTEPS_PER_INTERVAL = 2

# The Krylov basis of a matrix-free solve or exponential is bounded so that its
# memory stays independent of the 4**n state; a solve that needs more than this
# many vectors restarts instead of growing the basis.
SOLVE_RESTART = 30

_INTEGRATOR_ORDERS: Mapping[str, int | None] = {
    RUNGE_KUTTA: 4,
    CRANK_NICOLSON: 2,
    KRYLOV_EXPONENTIAL: None,
}

_SUBSTEPS: Mapping[str, int] = {
    RUNGE_KUTTA: SUBSTEPS_PER_INTERVAL,
    CRANK_NICOLSON: SUBSTEPS_PER_INTERVAL,
    # An exponential step is exact for a time-independent generator, so
    # subdividing its interval would repeat the same exact step and pay for it
    # twice.
    KRYLOV_EXPONENTIAL: 1,
}

INTEGRATOR_NAMES: tuple[str, ...] = tuple(_INTEGRATOR_ORDERS)

KRYLOV_SCHEMES: tuple[str, ...] = (CRANK_NICOLSON, KRYLOV_EXPONENTIAL)
"""Schemes whose accuracy is a Krylov precision target, not a step order."""


def integrator_order(method: str) -> int | None:
    """Return the step-size order of a scheme, or ``None`` if it has none.

    ``None`` is the declared answer for a scheme whose error does not fall as a
    power of the step size: the exponential of a time-independent generator is
    exact on the grid, so a finite order would understate it.
    """

    return _INTEGRATOR_ORDERS[method]


def substeps_per_interval(method: str) -> int:
    """Return how many internal steps a scheme takes across one output interval."""

    return _SUBSTEPS[method]


def advance_interval(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    interval: torch.Tensor,
    *,
    method: str,
    solve_tolerance: float,
    solve_max_iterations: int,
) -> tuple[torch.Tensor, float]:
    """Integrate ``state`` across one output interval with the named scheme.

    The second element of the return value is the relative defect of the last
    Krylov step -- the residual of the implicit solve, or the defect the
    exponential action measured -- or ``0.0`` when the scheme is explicit. The
    caller owns the policy for what defect is acceptable, so a scheme that could
    not reach its target is never silently reported as a converged evolution. An
    unknown method is refused here as well as at planning time; no scheme is
    ever substituted for another.
    """

    if method not in _INTEGRATOR_ORDERS:
        raise ValueError(f"unknown integration method {method!r}")
    step = interval / substeps_per_interval(method)
    if method == RUNGE_KUTTA:
        resolved = step.to(dtype=state.real.dtype)
        for _ in range(SUBSTEPS_PER_INTERVAL):
            state = _runge_kutta_4_step(derivative, state, resolved)
        return state, 0.0
    if method == CRANK_NICOLSON:
        return _crank_nicolson_interval(
            derivative,
            state,
            step.to(dtype=state.real.dtype),
            tolerance=solve_tolerance,
            max_iterations=solve_max_iterations,
        )
    return _krylov_exponential_interval(
        derivative,
        state,
        float(step),
        tolerance=solve_tolerance,
        max_iterations=solve_max_iterations,
    )


def _runge_kutta_4_step(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    step: torch.Tensor,
) -> torch.Tensor:
    """Take one classical fourth-order Runge--Kutta step."""

    k1 = derivative(state)
    k2 = derivative(state + step * k1 / 2)
    k3 = derivative(state + step * k2 / 2)
    k4 = derivative(state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _crank_nicolson_interval(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    step: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, float]:
    """Take the interval's trapezoidal substeps through matrix-free solves."""

    residual = 0.0
    for _ in range(SUBSTEPS_PER_INTERVAL):
        state, residual = _crank_nicolson_step(
            derivative,
            state,
            step,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    return state, residual


def _crank_nicolson_step(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    step: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, float]:
    """Take one second-order trapezoidal step through a matrix-free solve.

    A leading batch axis is folded into the solved vector: the generator acts
    on every member independently, so the flattened operator is block diagonal
    and one solve advances the whole batch. The reported residual is the
    relative defect of that flat vector.
    """

    shape = state.shape

    def apply(vector: torch.Tensor) -> torch.Tensor:
        matrix = vector.reshape(shape)
        return (matrix - (step / 2) * derivative(matrix)).reshape(-1)

    rhs = (state + (step / 2) * derivative(state)).reshape(-1)
    solution, residual = gmres(
        apply,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        restart=SOLVE_RESTART,
    )
    return solution.reshape(shape), residual


def _krylov_exponential_interval(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    step: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, float]:
    """Advance a whole interval by one matrix-free exponential step.

    The exponential acts on the vectorized state, and the generator is applied
    there through the density form: for the row-major vectorization
    ``Tensor.reshape(-1)`` gives, ``vec(d rho / dt)`` equals ``L vec(rho)``, so
    reshaping a Krylov vector into a density matrix, applying the density-form
    generator and reshaping back is an application of ``L`` that never forms it.
    A leading batch axis is folded into that vector, where the generator is
    block diagonal and one Krylov window advances every member.
    """

    shape = state.shape

    def apply(vector: torch.Tensor) -> torch.Tensor:
        matrix = vector.reshape(shape)
        return derivative(matrix).reshape(-1)

    advanced, relative = exponential_action(
        apply,
        step,
        state.reshape(-1),
        tolerance=tolerance,
        max_iterations=max_iterations,
        restart=SOLVE_RESTART,
    )
    return advanced.reshape(shape), relative


__all__ = (
    "CRANK_NICOLSON",
    "DEFAULT_INTEGRATOR",
    "INTEGRATOR_NAMES",
    "KRYLOV_EXPONENTIAL",
    "KRYLOV_SCHEMES",
    "RUNGE_KUTTA",
    "SOLVE_RESTART",
    "SUBSTEPS_PER_INTERVAL",
    "advance_interval",
    "integrator_order",
    "substeps_per_interval",
)
