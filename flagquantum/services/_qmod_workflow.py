"""Execute a checked declarative Qmod workflow with native algorithms."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

from ..algorithms._diagonal_objective import DiagonalMetrics, DiagonalQaoa
from ..algorithms.cobyla import CobylaOptimizer
from ..ecosystem._qmod.cscope import QaoaWorkflow


@dataclass(frozen=True)
class VqeEvaluation:
    parameters: tuple[float, ...]
    metrics: DiagonalMetrics


@dataclass(frozen=True)
class VqeRun:
    best: VqeEvaluation
    evaluations: tuple[VqeEvaluation, ...]
    termination: Literal["evaluation_limit", "optimizer_stopped"]
    statevector: torch.Tensor


@dataclass(frozen=True)
class QmodWorkflowResult:
    vqe: VqeRun
    saved: Mapping[str, VqeRun | tuple[tuple[str, float], ...]]


class _EvaluationLimitError(Exception):
    pass


def run_qaoa_workflow(
    workflow: QaoaWorkflow, *, memory_limit_bytes: int
) -> QmodWorkflowResult:
    """Run the bounded workflow, with save captured as named result values."""
    state_bytes = (1 << workflow.qubits) * 16
    if workflow.qubits > 24 or state_bytes > memory_limit_bytes:
        raise ValueError(
            f"QAOA workflow needs {state_bytes} bytes for its state and at most 24 qubits"
        )
    settings = workflow.settings
    optimizer = CobylaOptimizer(
        maxiter=settings.max_iteration,
        rhobeg=settings.step_size or 0.5,
        rhoend=settings.tolerance or 1e-8,
    )
    diagonal = DiagonalQaoa(
        workflow.qubits,
        workflow.diagonal,
        alpha=settings.alpha_cvar,
        maximize=settings.maximize,
        variance=not settings.skip_compute_variance,
    )
    observed: list[VqeEvaluation] = []
    best: VqeEvaluation | None = None
    state: torch.Tensor | None = None
    direction = -1 if settings.maximize else 1

    def objective(point: torch.Tensor) -> torch.Tensor:
        nonlocal best, state
        if len(observed) >= settings.max_iteration:
            raise _EvaluationLimitError
        parameters = tuple(float(item) for item in point.tolist())
        amplitudes = diagonal.state(parameters)
        evaluation = VqeEvaluation(parameters, diagonal.metrics(amplitudes))
        observed.append(evaluation)
        cost = direction * evaluation.metrics.energy
        if best is None or cost < direction * best.metrics.energy:
            best, state = evaluation, amplitudes
        return torch.tensor(cost, dtype=torch.float64)

    termination: Literal["evaluation_limit", "optimizer_stopped"] = "optimizer_stopped"
    try:
        optimizer.minimize(
            objective, torch.tensor(settings.initial_point, dtype=torch.float64)
        )
    except _EvaluationLimitError:
        termination = "evaluation_limit"
    assert best is not None and state is not None
    result = VqeRun(best, tuple(observed), termination, state)
    hamiltonian = tuple(
        (
            "".join(
                "Z" if mask & (1 << bit) else "I" for bit in range(workflow.qubits)
            ),
            coefficient,
        )
        for mask, coefficient in workflow.diagonal
    )
    saved = {
        name: result if kind == "result" else hamiltonian
        for name, kind in settings.saved
    }
    return QmodWorkflowResult(result, MappingProxyType(saved))
