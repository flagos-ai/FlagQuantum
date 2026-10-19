"""Deterministic continuous-time Lindblad evolution on dense states."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence  # noqa: F401
from dataclasses import dataclass
from typing import Any

import torch

from ..errors import ExecutionError, ValidationError  # noqa: F401
from ..observables import Observable  # noqa: F401
from .density_matrix import expand_operator  # noqa: F401
from .lindblad_generator import Liouvillian
from .lindblad_integrators import (
    DEFAULT_INTEGRATOR,
    INTEGRATOR_NAMES,
    KRYLOV_SCHEMES,
    SOLVE_RESTART,  # noqa: F401
    advance_interval,
    integrator_order,
    substeps_per_interval,
)

# Request normalization moved to ``lindblad_request``; the names below stay
# bound here because callers and the unit suite already reach them through this
# module path. A ``noqa`` marks a name only that path uses.
from .lindblad_request import (
    CollapseOperator,
    EvolutionValidationError,
    _as_complex_matrix,  # noqa: F401
    _descriptor_matrix,  # noqa: F401
    _descriptor_pauli,  # noqa: F401
    _error,
    _hamiltonian_terms,  # noqa: F401
    _named_operator,  # noqa: F401
    _normalize_collapse_operators,
    _normalize_collapse_terms,  # noqa: F401
    _normalize_hamiltonian,
    _normalize_initial_state,
    _normalize_observables,
    _normalize_times,
    _observable_matrix,  # noqa: F401
    _observable_terms,  # noqa: F401
    _pauli_operator,  # noqa: F401
    _pauli_term,  # noqa: F401
    _pauli_term_wires,  # noqa: F401
    _require_density_matrices,  # noqa: F401
    _require_gradients_flag,
    _require_hermitian_terms,  # noqa: F401
    _resolve_batch_size,
    _solve_max_iterations,
    _solve_tolerance,
    _statevectors_to_density,  # noqa: F401
)
from .matrix_free_hamiltonian import PauliSum, PauliSumTerm  # noqa: F401


def amplitude_damping(rate: float, qubit: int) -> CollapseOperator:
    """Return ``sqrt(rate) * |0><1|`` on one qubit.

    ``rate`` is a physical inverse-time rate, never a channel probability.
    """

    return CollapseOperator("amplitude_damping", rate, (qubit,))


@dataclass(frozen=True)
class EvolutionResult:
    """A time-resolved density-matrix evolution result.

    ``populations`` has shape ``(len(times), 2**n_qubits)``.  Observable values
    have one row per requested time.  Density matrices are retained only when
    ``return_density_matrices=True`` was requested.
    """

    times: torch.Tensor
    populations: torch.Tensor
    observables: dict[str, torch.Tensor]
    density_matrices: torch.Tensor | None
    maximum_trace_drift: float
    population_bounded: bool
    method: str
    order: int | None
    step_size: float
    precision: str
    population_tolerance: float
    trace_tolerance: float
    plan: object | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        """Return the time-resolved computational-basis probabilities."""

        return self.populations

    @property
    def expectations(self) -> tuple[torch.Tensor, ...]:
        """Return requested expectation trajectories in output order."""

        return tuple(self.observables.values())

    def expectation(self, selector: int | str | None = None) -> torch.Tensor:
        """Return one expectation trajectory by output index or name."""

        outputs = tuple(self.observables.items())
        if not outputs:
            raise ExecutionError("evolution result does not contain an expectation")
        if selector is None:
            if len(outputs) != 1:
                raise ExecutionError(
                    "evolution result contains multiple expectations; select one by "
                    "output index or name"
                )
            return outputs[0][1]
        if type(selector) is int:
            try:
                return outputs[selector][1]
            except IndexError as exc:
                raise ExecutionError(
                    f"expectation index {selector} is outside the result"
                ) from exc
        if not isinstance(selector, str) or not selector:
            raise TypeError(
                "expectation selector must be an integer or non-empty string"
            )
        try:
            return self.observables[selector]
        except KeyError as exc:
            raise ExecutionError(
                f"evolution result has no expectation {selector!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation, including provenance."""

        return {
            "times": self.times.detach().cpu().tolist(),
            "populations": self.populations.detach().cpu().tolist(),
            "observables": {
                name: values.detach().cpu().tolist()
                for name, values in self.observables.items()
            },
            "density_matrices": (
                None
                if self.density_matrices is None
                else _complex_tensor_to_json(self.density_matrices)
            ),
            "maximum_trace_drift": self.maximum_trace_drift,
            "population_bounded": self.population_bounded,
            "numerics": {
                "method": self.method,
                "order": self.order,
                "step_size": self.step_size,
                "precision": self.precision,
                "population_tolerance": self.population_tolerance,
                "trace_tolerance": self.trace_tolerance,
            },
            "plan_identity": getattr(self.plan, "identity", None),
        }


@dataclass(frozen=True)
class EvolutionPlan:
    """Validated resource and numerical-method plan for an evolution request."""

    n_wires: int
    n_times: int
    dimension: int
    trajectory_bytes: int
    method: str
    order: int | None
    step_size: float
    device: str
    precision: str
    solve_tolerance: float | None = None
    batch_size: int = 1
    require_gradients: bool = False

    @property
    def n_qubits(self) -> int:
        """Return the public qubit count; ``n_wires`` remains serialized internally."""

        return self.n_wires

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_wires": self.n_wires,
            "n_times": self.n_times,
            "dimension": self.dimension,
            "trajectory_bytes": self.trajectory_bytes,
            "batch_size": self.batch_size,
            "numerics": {
                "method": self.method,
                "order": self.order,
                "step_size": self.step_size,
                "device": self.device,
                "precision": self.precision,
                "solve_tolerance": self.solve_tolerance,
                "require_gradients": self.require_gradients,
            },
        }


def _complex_tensor_to_json(value: torch.Tensor) -> list[Any]:
    raw = value.detach().cpu()
    return torch.stack((raw.real, raw.imag), dim=-1).tolist()


def plan_density_matrix_evolution(
    hamiltonian: Any,
    initial_state: Any,
    n_qubits: int,
    times: Any,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> EvolutionPlan:
    """Validate an evolution request and report its deterministic local plan."""

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise _error(
            "invalid_n_qubits", "n_qubits must be a positive integer", "n_qubits"
        )
    if dtype not in {torch.complex64, torch.complex128}:
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
        )
    if method not in INTEGRATOR_NAMES:
        raise _error(
            "unsupported_method",
            "method must be one of "
            + ", ".join(repr(name) for name in INTEGRATOR_NAMES)
            + f", not {method!r}",
            "method",
        )
    resolved_require_gradients = _require_gradients_flag(require_gradients)
    resolved_solve_tolerance = _solve_tolerance(
        solve_tolerance, dtype=dtype, method=method
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise _error(
            "unsupported_device",
            "continuous-time evolution currently supports CPU only",
            "device",
        )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    dim = 2**n_qubits
    time_grid = _normalize_times(times, device=resolved_device, real_dtype=real_dtype)
    _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_qubits,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    initial_matrices = _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    resolved_batch_size = _resolve_batch_size(
        batch_size, 1 if initial_matrices.ndim == 2 else int(initial_matrices.shape[0])
    )
    _normalize_collapse_operators(
        collapse_operators,
        n_wires=n_qubits,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    _normalize_observables(observables, dim=dim, dtype=dtype, device=resolved_device)
    step_size = (
        0.0
        if time_grid.numel() == 1
        else float(
            (
                torch.max(time_grid[1:] - time_grid[:-1])
                / substeps_per_interval(method)
            ).item()
        )
    )
    element_bytes = 8 if dtype == torch.complex64 else 16
    return EvolutionPlan(
        n_wires=n_qubits,
        n_times=int(time_grid.numel()),
        dimension=dim,
        trajectory_bytes=(
            int(time_grid.numel()) * resolved_batch_size * dim * dim * element_bytes
        ),
        batch_size=resolved_batch_size,
        method=method,
        order=integrator_order(method),
        step_size=step_size,
        device=str(resolved_device),
        precision="complex64" if dtype == torch.complex64 else "complex128",
        solve_tolerance=(
            resolved_solve_tolerance if method in KRYLOV_SCHEMES else None
        ),
        require_gradients=resolved_require_gradients,
    )


def evolve_density_matrix(
    hamiltonian: Any,
    initial_state: Any,
    n_qubits: int,
    times: Any,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> EvolutionResult:
    """Evolve a state on an explicit time grid with the Lindblad equation.

    Collapse entries are either ``(operator, rate)`` pairs or mappings with
    ``operator`` (a dense matrix, ``"amplitude_damping"``, or
    ``"sigma_minus"``), ``rate``, and optional ``qubit``/``qubits``.  Rates,
    Hamiltonian coefficients, and time values must use the same inverse-time
    unit.  The named amplitude-damping operator is ``sqrt(rate) * |0><1|``;
    rates are never interpreted as channel probabilities.

    ``method="runge-kutta"`` takes two explicit fourth-order Runge--Kutta steps
    between every pair of output times.  ``method="crank-nicolson"`` takes two
    second-order trapezoidal steps that are solved matrix-free, so a stiff
    generator stays stable at a step where the explicit scheme would diverge;
    ``solve_tolerance`` sets the relative residual that solve must reach.
    ``method="krylov-exponential"`` advances each output interval by one
    matrix-free action of the generator's exponential, which has no step-size
    error for a time-independent generator, and reports ``order=None`` because
    its accuracy is set by ``solve_tolerance`` rather than by an order.

    An ``initial_state`` with a leading batch axis evolves every member under
    the same Hamiltonian, so one backward pass covers the whole batch. The
    trajectory then reports one population row per time and member. Every step
    is written with differentiable ``torch`` operations, so the returned
    populations stay connected to a differentiable ``initial_state`` and
    Hamiltonian. ``require_gradients`` asserts that connection and raises
    instead of returning a detached trajectory that still looks correct.

    Args:
        batch_size: The declared batch axis. ``None`` uses the batch the
            ``initial_state`` itself carries; a declared batch that disagrees
            with it is refused.
        require_gradients: Refuse to return a trajectory that carries no
            gradient graph. A plan deserialized from JSON cannot supply live
            tensors, so it fails here instead of silently detaching.

    Examples:
        >>> import torch
        >>> from flagquantum.simulation import amplitude_damping, evolve_density_matrix
        >>> times = torch.linspace(0.0, 1.0, 5)
        >>> result = evolve_density_matrix(
        ...     0.0 * torch.eye(2), [[0.0, 0.0], [0.0, 1.0]], 1, times,
        ...     [amplitude_damping(rate=2.0, qubit=0)], method="crank-nicolson",
        ... )
        >>> result.method, result.order
        ('crank-nicolson', 2)
    """

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise _error(
            "invalid_n_qubits", "n_qubits must be a positive integer", "n_qubits"
        )
    if dtype not in {torch.complex64, torch.complex128}:
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
        )
    if method not in INTEGRATOR_NAMES:
        raise _error(
            "unsupported_method",
            "method must be one of "
            + ", ".join(repr(name) for name in INTEGRATOR_NAMES)
            + f", not {method!r}",
            "method",
        )
    resolved_require_gradients = _require_gradients_flag(require_gradients)
    resolved_solve_tolerance = _solve_tolerance(
        solve_tolerance, dtype=dtype, method=method
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise _error(
            "unsupported_device",
            "continuous-time evolution currently supports CPU only",
            "device",
        )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    validation_tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    population_tolerance = 2e-5 if dtype == torch.complex64 else 1e-9
    trace_tolerance = 2e-5 if dtype == torch.complex64 else 1e-9
    dim = 2**n_qubits

    time_grid = _normalize_times(times, device=resolved_device, real_dtype=real_dtype)
    hamiltonian_representation = _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_qubits,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=validation_tolerance,
    )
    rho = _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=validation_tolerance,
    )
    # A declared batch must agree with the batch the initial state carries. The
    # resolved value is not needed afterwards, because every later stage reads
    # the axis off ``rho``; the refusal is the point of the call.
    _resolve_batch_size(batch_size, 1 if rho.ndim == 2 else int(rho.shape[0]))
    collapse = _normalize_collapse_operators(
        collapse_operators,
        n_wires=n_qubits,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    observable_matrices = _normalize_observables(
        observables, dim=dim, dtype=dtype, device=resolved_device
    )
    generator = Liouvillian(hamiltonian_representation, collapse, hilbert_dimension=dim)

    states = [rho]
    residuals = []
    for start, stop in zip(time_grid[:-1], time_grid[1:], strict=True):
        rho, residual = advance_interval(
            generator.derivative,
            rho,
            stop - start,
            method=method,
            solve_tolerance=resolved_solve_tolerance,
            solve_max_iterations=_solve_max_iterations(dim),
        )
        residuals.append(residual)
        states.append(rho)
    if residuals and max(residuals) > resolved_solve_tolerance:
        raise _error(
            "integration_not_converged",
            "the Krylov step did not reach the requested relative defect: "
            f"{max(residuals):.3e} > {resolved_solve_tolerance:.3e}",
            "solve_tolerance",
        )
    trajectory = torch.stack(states)
    # The requirement is on the returned trajectory, not on either input: a
    # caller may parameterize the Hamiltonian and hold a fixed initial state,
    # or the other way round. Only the trajectory knows whether the graph
    # survived, and a detached trajectory is indistinguishable by value, so
    # this is the one place the requirement can be decided.
    if resolved_require_gradients and not trajectory.requires_grad:
        raise _error(
            "gradients_not_available",
            "require_gradients was requested but the trajectory carries no "
            "gradient graph; a plan restores its request from JSON, which holds "
            "detached tensors",
            "require_gradients",
        )
    populations = torch.real(torch.diagonal(trajectory, dim1=-2, dim2=-1))
    traces = torch.diagonal(trajectory, dim1=-2, dim2=-1).sum(dim=-1)
    maximum_trace_drift = float(torch.max(torch.abs(traces - 1.0)).item())
    population_bounded = bool(
        (
            (populations >= -population_tolerance)
            & (populations <= 1.0 + population_tolerance)
        )
        .all()
        .item()
    )
    observable_values = {
        name: torch.real(
            torch.einsum(
                "tij,ji->t" if trajectory.ndim == 3 else "tbij,ji->tb",
                trajectory,
                matrix,
            )
        )
        for name, matrix in observable_matrices.items()
    }
    step_size = (
        0.0
        if time_grid.numel() == 1
        else float(
            (
                torch.max(time_grid[1:] - time_grid[:-1])
                / substeps_per_interval(method)
            ).item()
        )
    )
    return EvolutionResult(
        times=time_grid,
        populations=populations,
        observables=observable_values,
        density_matrices=trajectory if return_density_matrices else None,
        maximum_trace_drift=maximum_trace_drift,
        population_bounded=population_bounded,
        method=method,
        order=integrator_order(method),
        step_size=step_size,
        precision="complex64" if dtype == torch.complex64 else "complex128",
        population_tolerance=population_tolerance,
        trace_tolerance=trace_tolerance,
    )


__all__ = (
    "CollapseOperator",
    "EvolutionPlan",
    "EvolutionResult",
    "EvolutionValidationError",
    "amplitude_damping",
    "evolve_density_matrix",
    "plan_density_matrix_evolution",
)
