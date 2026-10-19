"""Continuous-time Lindblad evolution using FlagQuantum public contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch

from ..errors import CapabilityError
from ..observables import OutputRequest
from ..runtime.options import ExecutionOptions
from ..simulation.lindblad import (
    CollapseOperator,
    EvolutionResult,
    EvolutionValidationError,
    evolve_density_matrix,
)
from ..simulation.lindblad import (
    amplitude_damping as _amplitude_damping,
)
from ..simulation.lindblad_adjoint import (
    adjoint_gradient as _adjoint_gradient,
)
from ..simulation.lindblad_integrators import DEFAULT_INTEGRATOR, INTEGRATOR_NAMES
from ._plan import LindbladPlan, build_lindblad_plan

_MISSING = object()


def amplitude_damping(rate: float, qubit: int) -> CollapseOperator:
    """Return an amplitude-damping collapse operator with a physical rate."""

    return _amplitude_damping(rate, qubit)


def _infer_n_qubits(initial_state: Any) -> int:
    if isinstance(initial_state, str):
        if initial_state and not set(initial_state) - {"0", "1"}:
            return len(initial_state)
        raise EvolutionValidationError(
            "invalid_initial_state",
            "initial_state bitstring must contain binary digits",
            field="initial_state",
        )
    try:
        state = torch.as_tensor(initial_state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise EvolutionValidationError(
            "cannot_infer_n_qubits",
            "n_qubits could not be inferred from initial_state; pass n_qubits explicitly",
            field="n_qubits",
        ) from exc
    if state.ndim == 1 or (state.ndim == 2 and state.shape[0] == state.shape[1]):
        dimension = int(state.shape[0])
    elif state.ndim == 2:
        # A batch of statevectors: the trailing axis is the Hilbert dimension.
        dimension = int(state.shape[-1])
    elif state.ndim == 3 and state.shape[1] == state.shape[2]:
        dimension = int(state.shape[1])
    else:
        raise EvolutionValidationError(
            "cannot_infer_n_qubits",
            "n_qubits could not be inferred from initial_state; pass n_qubits explicitly",
            field="n_qubits",
        )
    n_qubits = int(math.log2(dimension)) if dimension > 0 else -1
    if n_qubits < 1 or 2**n_qubits != dimension:
        raise EvolutionValidationError(
            "cannot_infer_n_qubits",
            "initial_state dimension must be a positive power of two",
            field="n_qubits",
        )
    return n_qubits


def _execution_settings(
    options: ExecutionOptions | None,
) -> tuple[torch.device | str, torch.dtype, int | None, bool]:
    """Resolve the options this domain honours; the rest are refused by name.

    ``batch_size`` and ``require_gradients`` are resolved here rather than
    refused: evolution is written in differentiable ``torch`` operations, so a
    batch and a gradient graph are things this domain can honour.
    """

    if options is None:
        return "cpu", torch.complex128, None, False
    if not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an fq.ExecutionOptions instance or None")
    if options.mode not in {None, "density_matrix"}:
        raise CapabilityError(
            "Lindblad evolution requires options.mode='density_matrix'"
        )
    unsupported = {
        name: value
        for name, value in {
            "backend": options.backend,
            "target": options.target,
            "shots": options.shots,
            "seed": options.seed,
            "memory_limit_bytes": options.memory_limit_bytes,
            "allow_approximate": options.allow_approximate,
            "allow_backend_fallback": options.allow_backend_fallback,
        }.items()
        if value is not None
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise CapabilityError(f"Lindblad evolution does not support options: {names}")
    precision = options.precision or "complex128"
    dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[precision]
    return (
        options.device or "cpu",
        dtype,
        options.batch_size,
        bool(options.require_gradients),
    )


def _observables(
    outputs: OutputRequest | Sequence[OutputRequest] | None,
) -> dict[str, object] | None:
    if outputs is None:
        return None
    requests = (outputs,) if isinstance(outputs, OutputRequest) else tuple(outputs)
    if not requests or any(not isinstance(item, OutputRequest) for item in requests):
        raise TypeError("outputs must be an OutputRequest or a non-empty sequence")
    unsupported = tuple(item.kind for item in requests if item.kind != "expectation")
    if unsupported:
        kinds = ", ".join(sorted(set(unsupported)))
        raise CapabilityError(
            "Lindblad evolution always returns probabilities and currently supports "
            f"only expectation output requests, not: {kinds}"
        )
    names = tuple(item.name for item in requests if item.name is not None)
    if len(set(names)) != len(names):
        raise EvolutionValidationError(
            "duplicate_output_name",
            "output names must be unique",
            field="outputs",
        )
    return {
        item.name or f"expectation_{index}": item.observable
        for index, item in enumerate(requests)
    }


def run(
    hamiltonian_or_plan: Any,
    initial_state: Any = _MISSING,
    times: Any = _MISSING,
    *,
    collapse_operators: Sequence[Any] | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    n_qubits: int | None = None,
    options: ExecutionOptions | None = None,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
) -> EvolutionResult:
    """Run a Lindblad plan or plan and run a request in one call."""

    if isinstance(hamiltonian_or_plan, LindbladPlan):
        if (
            initial_state is not _MISSING
            or times is not _MISSING
            or collapse_operators is not None
            or outputs is not None
            or n_qubits is not None
            or options is not None
            or return_density_matrices
            or method != DEFAULT_INTEGRATOR
            or solve_tolerance is not None
        ):
            raise TypeError(
                "a LindbladPlan is closed to initial_state, times, and semantic overrides"
            )
        execution_plan = hamiltonian_or_plan
    else:
        if initial_state is _MISSING or times is _MISSING:
            raise TypeError("fql.run requires initial_state and times")
        execution_plan = plan(
            hamiltonian_or_plan,
            initial_state,
            times,
            collapse_operators=collapse_operators,
            outputs=outputs,
            n_qubits=n_qubits,
            options=options,
            return_density_matrices=return_density_matrices,
            method=method,
            solve_tolerance=solve_tolerance,
        )
    result = evolve_density_matrix(**execution_plan._execution_request())
    return replace(result, plan=execution_plan)


def plan(
    hamiltonian: Any,
    initial_state: Any,
    times: Any,
    *,
    collapse_operators: Sequence[Any] | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    n_qubits: int | None = None,
    options: ExecutionOptions | None = None,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
) -> LindbladPlan:
    """Build a sealed, serializable plan without running it.

    ``method`` selects the integration scheme and must be one of the names in
    ``flagquantum.lindblad.INTEGRATOR_NAMES``; ``solve_tolerance`` sets the
    relative residual a Krylov scheme must reach and is refused for an explicit
    one.

    ``options.batch_size`` declares the batch axis an ``initial_state`` carries
    and ``options.require_gradients`` requires the trajectory to stay connected
    to the caller's tensors; both are recorded in the plan, so a sealed plan
    states what it will return before it runs.

    Examples:
        >>> import torch
        >>> from flagquantum.lindblad import plan, amplitude_damping
        >>> p = plan(
        ...     torch.eye(2), [[1.0, 0.0], [0.0, 0.0]], torch.linspace(0, 1, 5),
        ...     collapse_operators=[amplitude_damping(rate=1.0, qubit=0)],
        ...     method="crank-nicolson",
        ... )
        >>> p.method, p.order
        ('crank-nicolson', 2)
    """

    device, dtype, batch_size, require_gradients = _execution_settings(options)
    resolved_n_qubits = _infer_n_qubits(initial_state) if n_qubits is None else n_qubits
    return build_lindblad_plan(
        hamiltonian,
        initial_state,
        resolved_n_qubits,
        times,
        collapse_operators,
        _observables(outputs),
        device=device,
        dtype=dtype,
        return_density_matrices=return_density_matrices,
        method=method,
        solve_tolerance=solve_tolerance,
        batch_size=batch_size,
        require_gradients=require_gradients,
    )


def adjoint_gradient(
    hamiltonian: Any,
    initial_state: Any,
    times: Any,
    cost: Callable[[torch.Tensor], torch.Tensor],
    *,
    collapse_operators: Sequence[Any] | None = None,
    n_qubits: int | None = None,
    options: ExecutionOptions | None = None,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
) -> torch.Tensor:
    """Return the gradient of ``cost`` with respect to the initial state.

    ``cost`` reads the whole trajectory -- one density matrix per time, stacked
    along a leading axis -- and returns a scalar. The gradient is propagated by
    the adjoint method, so the forward sweep records no graph and the reverse
    pass holds the trajectory instead of it; the value returned is the exact
    transpose of the same discretized scheme ``run`` would have applied, and it
    is the cotangent of the normalized initial density matrix rather than of
    ``initial_state`` as written.

    There is no ``outputs`` argument here: the cost is the readout, so naming a
    second one would be a second source of truth for what is being measured.

    ``options.require_gradients`` is refused, because it asks for a
    differentiable trajectory and this route deliberately produces none; use
    ``run`` when the gradient must come from autograd instead of the adjoint.

    Examples:
        >>> import torch
        >>> from flagquantum.lindblad import adjoint_gradient, amplitude_damping
        >>> gradient = adjoint_gradient(
        ...     torch.eye(2, dtype=torch.complex128),
        ...     [[1.0, 0.0], [0.0, 0.0]],
        ...     torch.linspace(0, 1, 5, dtype=torch.float64),
        ...     lambda trajectory: torch.real(trajectory[-1, 1, 1]),
        ...     collapse_operators=[amplitude_damping(rate=1.0, qubit=0)],
        ... )
        >>> gradient.shape
        torch.Size([2, 2])
    """

    resolved_n_qubits = _infer_n_qubits(initial_state) if n_qubits is None else n_qubits
    device, dtype, batch_size, require_gradients = _execution_settings(options)
    return _adjoint_gradient(
        hamiltonian,
        initial_state,
        resolved_n_qubits,
        times,
        cost,
        collapse_operators,
        device=device,
        dtype=dtype,
        method=method,
        solve_tolerance=solve_tolerance,
        batch_size=batch_size,
        require_gradients=require_gradients,
    )


__all__ = (
    "CollapseOperator",
    "EvolutionResult",
    "INTEGRATOR_NAMES",
    "LindbladPlan",
    "adjoint_gradient",
    "amplitude_damping",
    "plan",
    "run",
)
