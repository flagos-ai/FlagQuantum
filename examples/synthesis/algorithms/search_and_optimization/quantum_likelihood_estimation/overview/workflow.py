"""Run both original Bayesian update policies at their full proposal budgets."""

import math
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq

from .quantum import HYPOTHESES, RealArray, build_experiment, likelihoods

TRUE_HYPOTHESIS = 0
THRESHOLD = 0.99
STANDARD_LIMIT = 10000
OPTIMIZED_LIMIT = 100
ANNEAL_STEPS = 8
NEIGHBORS = 20
COOLING = 0.9
FIXED_CONTROLS = (1.2566, 0.5712, 0.7854, math.tau)
EPSILON = 1e-12


@dataclass(frozen=True)
class Update:
    """A measured outcome and its full Bayesian likelihood update."""

    controls: tuple[float, ...]
    outcome: int
    likelihood: tuple[float, ...]
    posterior: tuple[float, ...]


@dataclass(frozen=True)
class Workflow:
    """Actual experiment records from both source policies."""

    standard: tuple[Update, ...]
    optimized: tuple[Update, ...]


def entropy(weights: RealArray, proposals: RealArray) -> RealArray:
    """Compute the expected posterior entropy from actual batched statevectors."""
    joint = weights[:, None, None] * likelihoods(proposals)
    probabilities = joint.sum(axis=0)
    posterior = np.divide(
        joint,
        probabilities[None],
        out=np.zeros_like(joint),
        where=probabilities[None] > EPSILON,
    )
    logits = np.log2(np.where(posterior > EPSILON, posterior, 1.0))
    return np.asarray(-(joint * logits).sum(axis=(0, 2)), dtype=np.float64)


def anneal(weights: RealArray) -> RealArray:
    """Use the eight-step source schedule and all 20 candidates at each step."""
    rng = np.random.default_rng(1234)
    # The minimum nonzero gap of the four hypotheses is two.
    bounds = np.asarray(
        (math.pi, math.pi, math.tau, math.pi, math.tau), dtype=np.float64
    )
    current = rng.random(5) * bounds
    current_cost = float(entropy(weights, current[None])[0])
    best, best_cost = current.copy(), current_cost
    temperature = 1.0
    for _ in range(ANNEAL_STEPS):
        offsets = rng.uniform(-1, 1, (NEIGHBORS, 5)) * bounds / 2
        proposals = np.remainder(current + temperature * offsets, bounds)
        values = entropy(weights, proposals)
        index = int(np.argmin(values))
        cost = float(values[index])
        if cost < current_cost or rng.random() < math.exp(
            (current_cost - cost) / temperature
        ):
            current, current_cost = proposals[index], cost
        if current_cost < best_cost:
            best, best_cost = current.copy(), current_cost
        temperature *= COOLING
    return np.asarray(best, dtype=np.float64)


def particle_time(weights: RealArray, rng: np.random.Generator) -> float:
    """Evaluate the Frobenius distance between the two sampled hypotheses."""
    i, j = rng.choice(len(HYPOTHESES), 2, replace=False, p=weights)
    axis_1, scale_1 = HYPOTHESES[int(i)]
    axis_2, scale_2 = HYPOTHESES[int(j)]
    scale = abs(scale_1 - scale_2) if axis_1 == axis_2 else math.hypot(scale_1, scale_2)
    return 1 / (math.sqrt(2) * scale)


def _run(*, optimized: bool, observations: torch.Generator) -> tuple[Update, ...]:
    weights = np.full(4, 0.25, dtype=np.float64)
    rng = np.random.default_rng(1234)
    history = []
    for _ in range(OPTIMIZED_LIMIT if optimized else STANDARD_LIMIT):
        if weights[TRUE_HYPOTHESIS] > THRESHOLD:
            break
        if optimized:
            controls = anneal(weights)
        else:
            controls = np.asarray(
                (particle_time(weights, rng), *FIXED_CONTROLS), dtype=np.float64
            )
        binding = tuple(map(float, controls))
        actual = (
            fq.run(build_experiment(TRUE_HYPOTHESIS, binding)).statevector().flatten()
        )
        measured = torch.square(torch.abs(actual))
        predictions = likelihoods(controls[None])[:, 0, :]
        torch.testing.assert_close(
            measured, torch.as_tensor(predictions[TRUE_HYPOTHESIS]), atol=1e-12, rtol=0
        )
        outcome = int(torch.multinomial(measured, 1, generator=observations))
        distribution = predictions[:, outcome]
        updated = weights * distribution
        weights = np.asarray(updated / updated.sum(), dtype=np.float64)
        history.append(
            Update(
                binding,
                outcome,
                tuple(map(float, distribution)),
                tuple(map(float, weights)),
            )
        )
    return tuple(history)


def run_workflow() -> Workflow:
    """Use one reproducible local measurement stream for both source experiments."""
    observations = torch.Generator().manual_seed(7)
    return Workflow(
        _run(optimized=False, observations=observations),
        _run(optimized=True, observations=observations),
    )
