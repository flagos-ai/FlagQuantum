"""Public continuous-time Lindblad evolution API.

This namespace is the user-facing facade. Numerical kernels and validation
remain owned by :mod:`flagquantum.simulation`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..simulation.lindblad import (
    CollapseOperator,
    EvolutionPlan,
    EvolutionResult,
    EvolutionValidationError,
    amplitude_damping,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)


def _infer_n_wires(initial_state: Any) -> int:
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
            "cannot_infer_n_wires",
            "n_wires could not be inferred from initial_state; pass n_wires explicitly",
            field="n_wires",
        ) from exc
    if state.ndim == 1 or (state.ndim == 2 and state.shape[0] == state.shape[1]):
        dimension = int(state.shape[0])
    else:
        raise EvolutionValidationError(
            "cannot_infer_n_wires",
            "n_wires could not be inferred from initial_state; pass n_wires explicitly",
            field="n_wires",
        )
    n_wires = int(math.log2(dimension)) if dimension > 0 else -1
    if n_wires < 1 or 2**n_wires != dimension:
        raise EvolutionValidationError(
            "cannot_infer_n_wires",
            "initial_state dimension must be a positive power of two",
            field="n_wires",
        )
    return n_wires


def solve(
    hamiltonian: Any,
    initial_state: Any,
    times: Any,
    *,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    n_wires: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
) -> EvolutionResult:
    """Solve Lindblad evolution on an explicit time grid.

    ``n_wires`` is inferred from a basis-state bitstring, statevector, or
    density matrix when omitted.
    """

    return evolve_density_matrix(
        hamiltonian,
        initial_state,
        _infer_n_wires(initial_state) if n_wires is None else n_wires,
        times,
        collapse_operators,
        observables,
        device=device,
        dtype=dtype,
        return_density_matrices=return_density_matrices,
    )


def evolve(
    hamiltonian: Any,
    initial_state: Any,
    times: Any,
    *,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    n_wires: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
) -> EvolutionResult:
    """Alias of :func:`solve` for evolution-oriented code."""

    return solve(
        hamiltonian,
        initial_state,
        times,
        collapse_operators=collapse_operators,
        observables=observables,
        n_wires=n_wires,
        device=device,
        dtype=dtype,
        return_density_matrices=return_density_matrices,
    )


def plan(
    hamiltonian: Any,
    initial_state: Any,
    times: Any,
    *,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    n_wires: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
) -> EvolutionPlan:
    """Validate and size a Lindblad request without evolving it."""

    return plan_density_matrix_evolution(
        hamiltonian,
        initial_state,
        _infer_n_wires(initial_state) if n_wires is None else n_wires,
        times,
        collapse_operators,
        observables,
        device=device,
        dtype=dtype,
        return_density_matrices=return_density_matrices,
    )


__all__ = (
    "CollapseOperator",
    "EvolutionPlan",
    "EvolutionResult",
    "EvolutionValidationError",
    "amplitude_damping",
    "evolve",
    "plan",
    "solve",
)
