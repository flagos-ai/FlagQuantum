"""Run both source adaptive schedules and the explicit fixed VQE schedule."""

from dataclasses import dataclass

import torch

from flagquantum.algorithms.nelder_mead import NelderMeadOptimizer
from flagquantum.compiler import ResourceEstimate

from .model import INNER_BUDGET, OUTER_BUDGET, POOL, TOLERANCE, Model
from .quantum import build_program, state


@dataclass(frozen=True)
class Optimization:
    """Retain fitted angles, the actual objective trace, and circuit metrics."""

    indices: tuple[int, ...]
    angles: tuple[float, ...]
    energies: tuple[float, ...]
    iterations: int
    evaluations: int
    converged: bool
    energy: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class AdaptiveStep:
    """Record all raw source gradients and their scaling for one step."""

    gradient_norm: float
    gradients: tuple[float, ...]
    chosen_index: int | None
    fit: Optimization


@dataclass(frozen=True)
class AdaptiveRun:
    """The measured trace and the stop reason from the source rule."""

    steps: tuple[AdaptiveStep, ...]
    stop: str
    selections: tuple[int, ...]

    @property
    def best(self) -> AdaptiveStep:
        """Return the actual optimized record with the lowest energy."""
        return min(self.steps, key=lambda step: step.fit.energy)


def screen(
    model: Model, indices: tuple[int, ...], angles: tuple[float, ...]
) -> tuple[float, ...]:
    """Evaluate the entire ordered pool on the actual synthesized state."""
    amplitudes = state(build_program(indices, angles))
    return tuple(map(float, ((model.screening @ amplitudes) @ amplitudes.conj()).real))


def optimize(
    model: Model, indices: tuple[int, ...], initial: torch.Tensor
) -> Optimization:
    """Run the local 100-update budget from the source parameter initialization."""
    count = len(indices)
    history: list[float] = []

    def objective(parameter: torch.Tensor) -> torch.Tensor:
        program = build_program(indices, tuple(map(float, parameter[:count])))
        amplitudes = state(program)
        energy = torch.vdot(amplitudes, model.hamiltonian @ amplitudes).real
        history.append(float(energy))
        return energy

    result = NelderMeadOptimizer(maxiter=INNER_BUDGET, initial_step=0.2).minimize(
        objective, initial
    )
    angles = tuple(map(float, result.parameters))
    resources = build_program(indices, angles[:count]).report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("Expected the native circuit resources")
    return Optimization(
        indices,
        angles,
        tuple(history),
        result.iterations,
        result.evaluations,
        result.converged,
        result.value,
        resources,
    )


def _stop(scaled_norm: float, gradients: tuple[float, ...], iteration: int) -> str:
    if scaled_norm < TOLERANCE:
        return "gradient tolerance"
    if iteration + 1 == OUTER_BUDGET:
        return "outer budget"
    return "all gradients nonnegative" if min(gradients) >= 0 else ""


def adapt(model: Model, *, modified: bool) -> AdaptiveRun:
    """Preserve source argmin selection, repeats and the standard one-step lag."""
    selected = (10,) if modified else ()
    angles: tuple[float, ...] = ()
    steps: list[AdaptiveStep] = []
    for iteration in range(OUTER_BUDGET):
        if modified:
            initial = torch.linspace(0, 1, len(selected), dtype=torch.float64)
            fit = optimize(model, selected, initial)
            gradients = screen(model, selected, fit.angles)
        else:
            gradients = screen(model, selected, angles)
        norm = sum(gradient**2 for gradient in gradients) ** 0.5 / len(gradients)
        stop = _stop(norm, gradients, iteration)
        choice = min(range(len(gradients)), key=gradients.__getitem__)
        if modified:
            steps.append(AdaptiveStep(norm, gradients, None if stop else choice, fit))
            if not stop:
                selected = (*selected, choice)
        else:
            initial = torch.linspace(0, 1, len(selected) + 1, dtype=torch.float64)
            fit = optimize(model, selected, initial)
            selected = (*selected, choice)
            angles = fit.angles
            steps.append(AdaptiveStep(norm, gradients, choice, fit))
        if stop:
            return AdaptiveRun(tuple(steps), stop, selected)
    raise RuntimeError("Expected the source outer-loop stopping rule")


def fixed(model: Model, indices: tuple[int, ...] | None = None) -> Optimization:
    """Run the declared modular schedule, or the source captured selection schedule."""
    if indices is None:
        indices = (0, *(2 * index % len(POOL) for index in range(9)))
    if len(indices) != OUTER_BUDGET:
        raise ValueError("The source fixed comparison requires ten generators")
    return optimize(model, indices, torch.zeros(OUTER_BUDGET, dtype=torch.float64))
