"""Reverse-mode differentiation of a Lindblad trajectory by the adjoint state.

:func:`~.lindblad.evolve_density_matrix` is written in differentiable ``torch``
operations, so ``torch.autograd`` already returns the exact vector-Jacobian
product of a scalar cost with respect to the state a trajectory started from.
That graph, however, retains the intermediate of every Runge--Kutta stage, of
every Crank--Nicolson solve and of every Krylov window, so what it holds grows
with the output grid as well as with the state.

This module computes the same product a second way. The forward sweep is
integrated under ``torch.no_grad()``, so no part of it is retained, and the cost
gradient is then propagated backwards through the adjoint equation

    d lambda / dt = -L^dag lambda,

integrated on the same grid with the same scheme and the same generator, read
through the Hilbert--Schmidt adjoint that
:meth:`~.lindblad_generator.Liouvillian.adjoint_derivative` supplies. What the
reverse pass holds is the trajectory the forward pass already returned -- one
density matrix per output time -- instead of the forward graph.

The two routes differentiate the same discretized trajectory, so neither is an
independent reference for the other: they must agree on the step, and a
disagreement is a defect in one of them rather than a better derivative. They do
agree, and for a reason worth writing down rather than measuring alone. One
output interval of a linear scheme is a function of the generator by itself --
``f(step * L)`` for a polynomial ``f`` in the explicit case, a rational ``f``
whose numerator and denominator are functions of the same matrix in the implicit
case, and the exponential in the third -- so its transpose is ``f(step * L^dag)``,
which is exactly what the same scheme computes when it is given ``L^dag``. The
adjoint pass is therefore the exact transpose of the forward scheme and not a
second scheme of the same order. The remaining error is the one the forward pass
already carries: an Arnoldi window's defect, or a matrix-free solve that stopped
short, and both are governed by the same ``solve_tolerance`` and fail closed as
``integration_not_converged`` here exactly as they do forward.

What the adjoint buys is memory, and that is the only claim made for it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .lindblad import (
    EvolutionValidationError,
    _normalize_collapse_operators,
    _normalize_hamiltonian,
    _solve_max_iterations,
    _solve_tolerance,
    evolve_density_matrix,
)
from .lindblad_generator import Liouvillian
from .lindblad_integrators import DEFAULT_INTEGRATOR, advance_interval

__all__ = ("adjoint_gradient",)


def adjoint_gradient(
    hamiltonian: Any,
    initial_state: Any,
    n_qubits: int,
    times: Any,
    cost: Callable[[torch.Tensor], torch.Tensor],
    collapse_operators: Sequence[Any] | None = None,
    observables: Any = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> torch.Tensor:
    """Return ``d cost / d rho(t_0)`` for the normalized initial density matrix.

    ``cost`` maps the whole trajectory -- one density matrix per output time, with
    the shape ``evolve_density_matrix(..., return_density_matrices=True)`` returns
    -- to a scalar. The forward sweep runs without recording a graph and the
    returned cotangent is propagated by the adjoint method, so the memory the
    reverse pass needs is the trajectory rather than the forward graph.

    The result is the gradient with respect to the *normalized* ``rho(t_0)`` the
    route builds, not with respect to ``initial_state`` as written: a caller who
    parameterizes the initial state chains the returned cotangent through their
    own normalization, and a caller who supplies a bitstring has nothing to
    differentiate and receives the cotangent for the state that bitstring denotes.

    Args:
        cost: The scalar to differentiate, read from the stacked trajectory.
        require_gradients: Refused when true. The adjoint route returns a
            cotangent and deliberately keeps no graph, so a request for one is a
            different request rather than an option this route can honour.

    Examples:
        >>> import torch
        >>> from flagquantum.simulation import amplitude_damping, adjoint_gradient
        >>> times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        >>> rho0 = torch.tensor(
        ...     [[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128
        ... )
        >>> gradient = adjoint_gradient(
        ...     0.0 * torch.eye(2, dtype=torch.complex128),
        ...     rho0,
        ...     1,
        ...     times,
        ...     lambda trajectory: torch.real(trajectory[-1, 1, 1]),
        ...     [amplitude_damping(rate=2.0, wire=0)],
        ...     dtype=torch.complex128,
        ... )
        >>> gradient.shape
        torch.Size([2, 2])
    """

    if require_gradients:
        raise EvolutionValidationError(
            "adjoint_requires_no_graph",
            "the adjoint route returns a cotangent for the initial state instead "
            "of a differentiable trajectory; require_gradients is not an option "
            "it can honour",
            field="require_gradients",
        )
    if not callable(cost):
        raise EvolutionValidationError(
            "invalid_cost",
            "cost must be callable on the trajectory of density matrices",
            field="cost",
        )
    resolved_device = torch.device(device)
    dim = 2**n_qubits
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    # The forward sweep reuses the engine, so every input is validated once and
    # in one place. ``no_grad`` is what makes the memory claim structural rather
    # than a hope: nothing about the sweep can be retained if nothing is
    # recorded.
    with torch.no_grad():
        forward = evolve_density_matrix(
            hamiltonian,
            initial_state,
            n_qubits,
            times,
            collapse_operators,
            observables,
            device=resolved_device,
            dtype=dtype,
            return_density_matrices=True,
            method=method,
            solve_tolerance=solve_tolerance,
            batch_size=batch_size,
        )
    trajectory = forward.density_matrices
    if trajectory is None:  # pragma: no cover - asked for above
        raise EvolutionValidationError(
            "missing_trajectory",
            "the forward sweep did not return its density matrices",
            field="times",
        )
    # The cost is differentiated on the trajectory alone. Its graph is the
    # caller's readout, which is typically a few reductions, so this is the part
    # that is allowed to be recorded -- and it is recorded explicitly, because a
    # caller who is already inside ``no_grad`` still wants a cotangent.
    rows = [row.detach().requires_grad_(True) for row in trajectory]
    with torch.enable_grad():
        value = cost(torch.stack(rows))
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise EvolutionValidationError(
            "non_scalar_cost",
            "cost must return a zero-dimensional tensor",
            field="cost",
        )
    seeds = torch.autograd.grad(value, rows, allow_unused=True, materialize_grads=True)
    # The generator is rebuilt from the same normalizers the forward sweep used,
    # so the backward pass differentiates the operator that was actually
    # integrated rather than a second reading of the caller's arguments. It is
    # assembled without a graph for the same reason the sweep is: a caller who
    # parameterized the Hamiltonian wants the cotangent, which is the seed of
    # their own graph, and not a second graph assembled from their leaf. Measured
    # on this fixture with a Hamiltonian leaf, recording this construction is the
    # only thing the route retains -- 2 tensors -- and skipping it makes the
    # route hold nothing in that case exactly as it holds nothing when the
    # Hamiltonian is a constant.
    with torch.no_grad():
        generator = Liouvillian(
            _normalize_hamiltonian(
                hamiltonian,
                n_wires=n_qubits,
                dim=dim,
                dtype=dtype,
                device=resolved_device,
                tolerance=tolerance,
            ),
            _normalize_collapse_operators(
                collapse_operators,
                n_wires=n_qubits,
                dim=dim,
                dtype=dtype,
                device=resolved_device,
            ),
            hilbert_dimension=dim,
        )
    resolved_tolerance = _solve_tolerance(solve_tolerance, dtype=dtype, method=method)
    max_iterations = _solve_max_iterations(dim)
    grid = forward.times
    residuals = []
    # No part of the reverse pass is recorded either, so the cotangent that
    # comes back is a value and not a second graph to hold -- including when the
    # caller parameterized the Hamiltonian, whose generator is differentiable.
    # A caller who wants that derivative takes the cotangent as the seed of
    # their own graph rather than the other way round.
    with torch.no_grad():
        cotangent = seeds[-1]
        for index in range(len(rows) - 1, 0, -1):
            # The same scheme over the same positive interval, applied to the
            # adjoint generator: the interval is walked backwards while the
            # propagator of each interval is the one the forward sweep applied.
            advanced, residual = advance_interval(
                generator.adjoint_derivative,
                cotangent,
                grid[index] - grid[index - 1],
                method=method,
                solve_tolerance=resolved_tolerance,
                solve_max_iterations=max_iterations,
            )
            residuals.append(residual)
            # Every output time contributes its own seed, not only the last one:
            # a cost that reads the trajectory at several times has a term at
            # each of them, and dropping this sum differentiates the final time
            # alone.
            cotangent = seeds[index - 1] + advanced
    if residuals and max(residuals) > resolved_tolerance:
        raise EvolutionValidationError(
            "integration_not_converged",
            "an adjoint Krylov step did not reach the requested relative defect: "
            f"{max(residuals):.3e} > {resolved_tolerance:.3e}",
            field="solve_tolerance",
        )
    return cotangent
