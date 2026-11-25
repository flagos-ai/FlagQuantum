"""Bound the optimizer orchestration separately from full quantum acceptance."""

import importlib
import math

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name",
    ("max_induced_k_color_subgraph", "min_graph_coloring"),
)
def test_two_iteration_trace_sampling_and_feasibility(name, monkeypatch):
    module = importlib.import_module(
        f"examples.synthesis.applications.optimization.{name}.overview.workflow"
    )
    feasible = min(
        (key for key in module.DOMAIN if module.model_row(key).feasible),
        key=lambda key: module.model_row(key).retained_quantum_cost,
    )
    infeasible = max(
        module.DOMAIN, key=lambda key: module.model_row(key).retained_quantum_cost
    )
    good_index = module.DOMAIN.index(feasible)
    bad_index = module.DOMAIN.index(infeasible)
    builds = []

    # Real parameter/state equivalence is checked in the companion modules.
    # This explicit quantum-reader double isolates the real optimizer, trace,
    # sampling and classical feasibility path without compiling every vertex.
    def build(parameters):
        parameters = tuple(parameters)
        builds.append(parameters)
        return parameters

    def read(parameters):
        mass = 0.5 + 0.2 * math.tanh(-sum((p - 0.25) ** 2 for p in parameters))
        probabilities = [0.0] * len(module.DOMAIN)
        probabilities[good_index] = mass
        probabilities[bad_index] = 1.0 - mass
        return tuple(probabilities)

    monkeypatch.setattr(module, "build_program", build)
    monkeypatch.setattr(module, "read_distribution", read)
    result = module.run_workflow(maxiter=2, shots=128, seed=2718)

    assert result.optimizer.iterations == 2
    assert not result.optimizer.converged
    assert result.optimizer.evaluations == len(result.trace)
    assert len(builds) == result.optimizer.evaluations + 1
    assert result.trace[0].parameters == module.DEFAULT_PARAMS
    assert result.optimizer.value <= result.trace[0].cost
    assert sum(row.count for row in result.samples) == 128
    assert sum(row.count for row in result.uniform) == 128
    assert result.samples == module.sample_distribution(
        read(result.program), shots=128, seed=2718
    )
    assert result.best_feasible is not None
    assert result.best_feasible.assignment == feasible
    for row in (*result.samples, *result.uniform):
        assert row == module.model_row(row.assignment, row.count)
