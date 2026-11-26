"""Variational eigensolver workflows and results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from importlib import import_module as _import_public_namespace
from time import perf_counter
from typing import Protocol

import torch

from flagquantum.algorithms.optimization import (
    HybridOptimizationResult,
    OptimizationStage,
    optimize_hybrid,
)
from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian
from flagquantum.circuit import Circuit


@dataclass(frozen=True)
class VQEResult:
    """Result returned by the native VQE optimizer."""

    parameters: torch.Tensor
    energy: torch.Tensor
    history: tuple[float, ...]

    @property
    def n_steps(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class AdaptVQEIteration:
    """One operator-selection and variational re-optimization stage."""

    selected_pool_index: int
    selected_gradient: float
    pool_gradients: tuple[float, ...]
    energy_before: float
    energy_after: float
    optimization_history: tuple[float, ...]
    screening_seconds: float
    optimization_seconds: float


@dataclass(frozen=True)
class AdaptVQEResult:
    """Result of a deterministic PyTorch-native ADAPT-VQE growth loop."""

    selected_pool_indices: tuple[int, ...]
    parameters: torch.Tensor
    energy: torch.Tensor
    initial_energy: float
    iterations: tuple[AdaptVQEIteration, ...]
    converged: bool

    @property
    def n_adapt_iterations(self) -> int:
        return len(self.iterations)


def vqe_loss(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    parameters: torch.Tensor,
    hamiltonian: Hamiltonian,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate a differentiable VQE objective."""

    values = hamiltonian.expectation(circuit_builder(parameters))
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


class OptimizerFactory(Protocol):
    """Construct an optimizer from trainable tensors and a learning rate."""

    def __call__(
        self, params: Iterable[torch.Tensor], *, lr: float
    ) -> torch.optim.Optimizer: ...


def _require_parameter_dependence(
    objective: torch.Tensor, *, owner: str, cause: str
) -> None:
    """Refuse an objective that carries no gradient, before torch raises for us.

    Every optimization loop in this module differentiates its objective with
    respect to the parameters it updates, so an objective that does not depend on
    those parameters is a caller error rather than a step of a descent. Torch
    reports it as ``element 0 of tensors does not require grad and does not have
    a grad_fn``, which names neither this run nor the builder that produced it, so
    the condition is checked here and named. Nothing about a run that does
    differentiate changes.
    """

    if objective.requires_grad:
        return
    raise ValueError(
        f"{owner}: the objective carries no gradient with respect to the "
        f"parameters, because {cause}. This loop differentiates the objective with "
        "respect to the parameters it updates, so an objective that does not depend "
        "on them cannot be descended."
    )


def run_vqe(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    steps: int = 100,
    lr: float = 0.05,
    optimizer_factory: OptimizerFactory = torch.optim.Adam,
) -> VQEResult:
    """Run a compact PyTorch-native VQE optimization loop."""

    parameters = torch.as_tensor(initial_parameters, dtype=torch.float32).clone()
    parameters = parameters.detach().requires_grad_(True)
    optimizer = optimizer_factory([parameters], lr=lr)
    history: list[float] = []

    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = vqe_loss(circuit_builder, parameters, hamiltonian)
        _require_parameter_dependence(
            loss,
            owner="run_vqe",
            cause="the circuit that circuit_builder returns does not depend on the "
            "parameters it was handed",
        )
        torch.autograd.backward(loss)
        optimizer.step()
        history.append(float(loss.detach()))

    final_energy = vqe_loss(circuit_builder, parameters, hamiltonian).detach()
    return VQEResult(
        parameters=parameters.detach().clone(),
        energy=final_energy,
        history=tuple(history),
    )


def run_adapt_vqe(
    circuit_builder: Callable[[tuple[object, ...], torch.Tensor], Circuit],
    operator_pool: Sequence[object],
    hamiltonian: Hamiltonian | None = None,
    *,
    energy_function: Callable[[Circuit], torch.Tensor] | None = None,
    screening_function: (
        Callable[
            [tuple[object, ...], torch.Tensor, tuple[object, ...]], Sequence[float]
        ]
        | None
    ) = None,
    max_adapt_iterations: int = 8,
    optimization_steps: int = 50,
    lr: float = 0.05,
    gradient_tolerance: float = 1e-6,
    optimizer_factory: OptimizerFactory = torch.optim.Adam,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> AdaptVQEResult:
    """Grow and optimize an ansatz using exact candidate gradients.

    ``circuit_builder`` receives the selected operator records and one scalar
    parameter per selected operator. Candidate operators are screened by
    appending them at zero angle and differentiating the exact Hamiltonian
    expectation. The selected operator is then retained and all accumulated
    parameters are re-optimized. ``screening_function`` may provide an exact,
    workload-specific gradient evaluator; it receives the current operators,
    parameters, and the currently available pool operators in that order.
    """

    pool = tuple(operator_pool)
    if not pool:
        raise ValueError("ADAPT-VQE requires a non-empty operator pool")
    if max_adapt_iterations <= 0:
        raise ValueError("max_adapt_iterations must be positive")
    if optimization_steps <= 0:
        raise ValueError("optimization_steps must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("ADAPT-VQE parameters require a floating dtype")
    tolerance = float(gradient_tolerance)
    if tolerance < 0.0:
        raise ValueError("gradient_tolerance must be non-negative")
    if (hamiltonian is None) == (energy_function is None):
        raise ValueError("provide exactly one of hamiltonian or energy_function")
    if energy_function is not None:
        evaluate_energy = energy_function
    else:
        assert hamiltonian is not None
        evaluate_energy = hamiltonian.expectation

    selected_indices: list[int] = []
    selected_operators: list[object] = []
    parameters = torch.empty(0, dtype=dtype, device=device)

    def energy(operators: tuple[object, ...], values: torch.Tensor) -> torch.Tensor:
        circuit = circuit_builder(operators, values)
        return evaluate_energy(circuit).sum()

    initial_energy = float(energy((), parameters).detach())
    records: list[AdaptVQEIteration] = []
    converged = False
    for _ in range(int(max_adapt_iterations)):
        screening_started = perf_counter()
        available = [
            pool_index
            for pool_index in range(len(pool))
            if pool_index not in selected_indices
        ]
        if not available:
            converged = True
            break
        if screening_function is None:
            gradients: list[float] = []
            for pool_index in available:
                candidate_values = torch.cat(
                    (parameters.detach(), parameters.new_zeros(1))
                ).requires_grad_(True)
                candidate_energy = energy(
                    (*selected_operators, pool[pool_index]), candidate_values
                )
                _require_parameter_dependence(
                    candidate_energy,
                    owner="run_adapt_vqe",
                    cause=(
                        f"the operator at pool index {pool_index} adds no dependence "
                        "on the angle it is appended at, so it is not a rotation or "
                        "the circuit builder does not apply it"
                    ),
                )
                candidate_gradient = torch.autograd.grad(
                    candidate_energy, candidate_values
                )[0][-1]
                gradients.append(float(candidate_gradient.detach()))
        else:
            evaluated = tuple(
                screening_function(
                    tuple(selected_operators),
                    parameters.detach(),
                    tuple(pool[index] for index in available),
                )
            )
            if len(evaluated) != len(available):
                raise ValueError(
                    "screening_function must return one gradient per candidate"
                )
            gradients = [float(gradient) for gradient in evaluated]
        best_position = max(
            range(len(available)), key=lambda index: abs(gradients[index])
        )
        best_index = available[best_position]
        best_gradient = gradients[best_position]
        if abs(best_gradient) <= tolerance:
            converged = True
            break
        screening_seconds = perf_counter() - screening_started

        energy_before = float(energy(tuple(selected_operators), parameters).detach())
        selected_indices.append(best_index)
        selected_operators.append(pool[best_index])
        parameters = torch.cat(
            (parameters.detach(), parameters.new_zeros(1))
        ).requires_grad_(True)
        optimizer = optimizer_factory([parameters], lr=lr)
        history: list[float] = []
        optimization_started = perf_counter()
        for _ in range(int(optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = energy(tuple(selected_operators), parameters)
            _require_parameter_dependence(
                loss,
                owner="run_adapt_vqe",
                cause="the circuit builder does not apply every selected operator, "
                "so the accumulated parameters do not all reach the energy",
            )
            torch.autograd.backward(loss)
            optimizer.step()
            history.append(float(loss.detach()))
        optimization_seconds = perf_counter() - optimization_started
        energy_after = float(energy(tuple(selected_operators), parameters).detach())
        full_gradients = [0.0] * len(pool)
        for pool_index, gradient in zip(available, gradients, strict=True):
            full_gradients[pool_index] = gradient
        records.append(
            AdaptVQEIteration(
                selected_pool_index=best_index,
                selected_gradient=best_gradient,
                pool_gradients=tuple(full_gradients),
                energy_before=energy_before,
                energy_after=energy_after,
                optimization_history=tuple(history),
                screening_seconds=screening_seconds,
                optimization_seconds=optimization_seconds,
            )
        )

    final_energy = energy(tuple(selected_operators), parameters).detach()
    return AdaptVQEResult(
        selected_pool_indices=tuple(selected_indices),
        parameters=parameters.detach().clone(),
        energy=final_energy,
        initial_energy=initial_energy,
        iterations=tuple(records),
        converged=converged,
    )


def run_hybrid_vqe(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    stages: Sequence[OptimizationStage],
) -> HybridOptimizationResult:
    """Run a staged VQE schedule using classical and quantum-aware methods.

    The parameter group is named ``"quantum"``.  Use :func:`optimize_hybrid`
    directly for models with both classical and quantum parameter groups.
    QNG evaluates the exact local statevector metric and is therefore intended
    for small-system convergence work, not distributed scalability claims.
    """

    initial = torch.as_tensor(initial_parameters)
    if not (initial.is_floating_point() or initial.is_complex()):
        initial = initial.to(torch.get_default_dtype())

    def objective(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return vqe_loss(circuit_builder, groups["quantum"], hamiltonian)

    def state_function(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return circuit_builder(groups["quantum"]).state().reshape(-1)

    return optimize_hybrid(
        objective,
        {"quantum": initial},
        stages,
        state_function=state_function,
    )


@dataclass(frozen=True)
class LayerwiseVQEResult:
    """VQE results produced while growing an ansatz depth by depth."""

    depths: tuple[int, ...]
    stages: tuple[HybridOptimizationResult, ...]

    @property
    def parameters(self) -> torch.Tensor:
        return self.stages[-1].parameters["quantum"]

    @property
    def energy(self) -> torch.Tensor:
        return self.stages[-1].energy


def run_layerwise_vqe(
    circuit_builder: Callable[[int, torch.Tensor], Circuit],
    parameter_count: Callable[[int], int],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    depths: Sequence[int],
    stages: Sequence[OptimizationStage],
) -> LayerwiseVQEResult:
    """Grow a variational circuit while preserving optimized earlier layers.

    New coordinates are initialized to zero, so Pauli-rotation layers begin as
    identity operations. This continuation path avoids restarting every deeper
    ansatz from an unrelated random point.
    """

    normalized_depths = tuple(int(depth) for depth in depths)
    if not normalized_depths or any(depth <= 0 for depth in normalized_depths):
        raise ValueError("depths must contain positive integers")
    if tuple(sorted(set(normalized_depths))) != normalized_depths:
        raise ValueError("depths must be strictly increasing")
    parameters = torch.as_tensor(initial_parameters).reshape(-1)
    if not (parameters.is_floating_point() or parameters.is_complex()):
        parameters = parameters.to(torch.get_default_dtype())
    first_count = int(parameter_count(normalized_depths[0]))
    if parameters.numel() != first_count:
        raise ValueError(
            f"initial parameters contain {parameters.numel()} values; expected {first_count}"
        )

    results: list[HybridOptimizationResult] = []
    for depth in normalized_depths:
        required = int(parameter_count(depth))
        if required < parameters.numel():
            raise ValueError("parameter_count must not shrink as depth increases")
        if required > parameters.numel():
            parameters = torch.cat(
                (parameters, parameters.new_zeros(required - parameters.numel()))
            )
        result = run_hybrid_vqe(
            partial(circuit_builder, depth),
            parameters,
            hamiltonian,
            stages=stages,
        )
        results.append(result)
        parameters = result.parameters["quantum"]
    return LayerwiseVQEResult(depths=normalized_depths, stages=tuple(results))


__all__ = (
    "VQEResult",
    "AdaptVQEIteration",
    "AdaptVQEResult",
    "vqe_loss",
    "OptimizerFactory",
    "run_vqe",
    "run_adapt_vqe",
    "run_hybrid_vqe",
    "LayerwiseVQEResult",
    "run_layerwise_vqe",
)

# Preserve the public algorithm and serialized result identities.
for _identity_export in (
    VQEResult,
    AdaptVQEIteration,
    AdaptVQEResult,
    vqe_loss,
    OptimizerFactory,
    _require_parameter_dependence,
    run_vqe,
    run_adapt_vqe,
    run_hybrid_vqe,
    LayerwiseVQEResult,
    run_layerwise_vqe,
):
    _identity_export.__module__ = "flagquantum.algorithms.core"
del _identity_export

# Bind the historical public namespace for class annotation introspection.

_public_namespace = _import_public_namespace("flagquantum.algorithms.core")
for _public_name in (
    "AdaptVQEIteration",
    "AdaptVQEResult",
    "LayerwiseVQEResult",
    "OptimizerFactory",
    "VQEResult",
    "_require_parameter_dependence",
    "run_adapt_vqe",
    "run_hybrid_vqe",
    "run_layerwise_vqe",
    "run_vqe",
    "vqe_loss",
):
    setattr(_public_namespace, _public_name, globals()[_public_name])
del _public_namespace, _public_name
