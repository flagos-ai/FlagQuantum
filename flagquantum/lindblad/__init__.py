"""Continuous-time Lindblad evolution using FlagQuantum public contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
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
) -> tuple[torch.device | str, torch.dtype]:
    if options is None:
        return "cpu", torch.complex128
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
            "batch_size": options.batch_size,
            "shots": options.shots,
            "seed": options.seed,
            "memory_limit_bytes": options.memory_limit_bytes,
            "require_gradients": options.require_gradients,
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
    return options.device or "cpu", dtype


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
) -> LindbladPlan:
    """Build a sealed, serializable plan without running it."""

    device, dtype = _execution_settings(options)
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
    )


__all__ = (
    "CollapseOperator",
    "EvolutionResult",
    "LindbladPlan",
    "amplitude_damping",
    "plan",
    "run",
)
