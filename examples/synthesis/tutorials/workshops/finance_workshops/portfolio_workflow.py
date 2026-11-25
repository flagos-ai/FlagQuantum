"""Bounded local cost search and joint readout for both portfolio workshops."""

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal

import torch

import flagquantum as fq
from examples.synthesis.factor_reference import factored_cost_circuit
from examples.synthesis.portfolio_reference import (
    COVARIANCE,
    RETURNS,
    portfolio_cost,
    portfolio_factors,
)
from examples.synthesis.quadratic_reference import order_register_factors
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.compiler import ResourceEstimate

from .portfolio_programs import holdings_program, slack_program

if TYPE_CHECKING:
    from matplotlib.figure import Figure

Assignment = tuple[int, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    layers: int
    return_scale: float
    penalty: float
    evaluation_budget: int
    constraint: Literal["none", "equality", "inequality"]

    def __post_init__(self) -> None:
        if type(self.layers) is not int or self.layers not in (3, 4):
            raise ValueError("workshop scenarios require three or four layers")
        if (
            type(self.evaluation_budget) is not int
            or not 1 <= self.evaluation_budget <= 200
        ):
            raise ValueError("evaluation budget must lie in 1..200")
        if any(
            isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in (self.return_scale, self.penalty)
        ):
            raise ValueError("return scale and penalty must be finite and nonnegative")
        if self.constraint not in ("none", "equality", "inequality"):
            raise ValueError("unknown portfolio constraint")
        if self.constraint == "none" and self.penalty != 0:
            raise ValueError("unconstrained scenarios cannot apply a budget penalty")

    @property
    def registers(self) -> tuple[str, ...]:
        return (
            ("a0", "a1", "a2", "slack")
            if self.constraint == "inequality"
            else ("a0", "a1", "a2")
        )

    @property
    def width(self) -> int:
        return 9 if self.constraint == "inequality" else 6


EQUALITY_SCENARIOS = (
    Scenario("unconstrained introduction", 4, 1.5, 0.0, 30, "none"),
    Scenario("unconstrained complete solution", 3, 1.7, 0.0, 10, "none"),
    Scenario("equality penalty solution", 3, 10.0, 2.0, 40, "equality"),
)
INEQUALITY_SCENARIOS = (
    Scenario("inequality introduction", 4, 1.4, 1.3, 20, "inequality"),
    Scenario("inequality complete solution", 3, 10.0, 30.5, 20, "inequality"),
)


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    assignment: Assignment
    cost: float
    count: int
    probability: float
    budget_feasible: bool
    zero_penalty: bool

    @property
    def holdings(self) -> tuple[int, int, int]:
        return self.assignment[0], self.assignment[1], self.assignment[2]

    @property
    def slack(self) -> int | None:
        return self.assignment[3] if len(self.assignment) == 4 else None


@dataclass(frozen=True)
class Workflow:
    scenario: Scenario
    program: synth.Program
    trace: tuple[Evaluation, ...]
    selected: Evaluation
    samples: tuple[Sample, ...]
    termination: Literal["evaluation_budget", "optimizer_returned"]
    converged: bool

    @property
    def best_observed(self) -> Sample:
        return self.samples[0]


@dataclass(frozen=True)
class StateCheck:
    parameters: tuple[float, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


def initial_parameters(scenario: Scenario) -> tuple[float, ...]:
    """Interleave the original linear gamma 0..pi and beta pi..0 schedules."""
    return tuple(
        value
        for i in range(scenario.layers)
        for value in (
            math.pi * i / (scenario.layers - 1),
            math.pi * (1 - i / (scenario.layers - 1)),
        )
    )


def bindings(scenario: Scenario) -> tuple[tuple[float, ...], ...]:
    return (
        initial_parameters(scenario),
        tuple(
            (i + 1) * 0.013 * (-1 if i % 2 else 1) for i in range(2 * scenario.layers)
        ),
        tuple(
            (i + 1) * 0.019 * (-1 if i % 3 else 1) for i in range(2 * scenario.layers)
        ),
    )


def build_program(scenario: Scenario, parameters: tuple[float, ...]) -> synth.Program:
    """Bind a source scenario without physical wire numbers or manual decoding."""
    if len(parameters) != 2 * scenario.layers or any(
        isinstance(v, bool) or not math.isfinite(v) for v in parameters
    ):
        raise ValueError("provide two finite radian parameters per layer")
    function = (
        slack_program if scenario.constraint == "inequality" else holdings_program
    )
    return synth.synthesize(
        function,
        classical_inputs={
            "params": parameters,
            "layers": scenario.layers,
            "return_scale": scenario.return_scale,
            "penalty": scenario.penalty,
        },
        max_qubits=scenario.width,
        max_workspace=0,
        max_operations=2048,
        memory_limit_bytes=4 * 1024**2,
    )


def risk(holdings: tuple[int, int, int]) -> float:
    return sum(
        COVARIANCE[i][j] * holdings[i] * holdings[j] for i in range(3) for j in range(3)
    )


def expected_return(holdings: tuple[int, int, int]) -> float:
    return sum(rate * value for rate, value in zip(RETURNS, holdings, strict=True))


def cost(scenario: Scenario, assignment: Assignment) -> float:
    """Evaluate joint asset/slack cost before any marginalization or filtering."""
    if len(assignment) != len(scenario.registers) or any(
        type(value) is not int or not 0 <= value < (8 if i == 3 else 4)
        for i, value in enumerate(assignment)
    ):
        raise ValueError("assignment must match the two-bit assets and numeric slack")
    holdings = assignment[0], assignment[1], assignment[2]
    slack = float(assignment[3]) if len(assignment) == 4 else 0.0
    return portfolio_cost(holdings, slack, scenario.penalty, scenario.return_scale, 1.0)


def assignments(scenario: Scenario) -> tuple[Assignment, ...]:
    domains = (range(4), range(4), range(4))
    return (
        tuple(product(*domains, range(8)))
        if scenario.constraint == "inequality"
        else tuple(product(*domains))
    )


def classical_optima(scenario: Scenario) -> tuple[tuple[Assignment, float], ...]:
    """Enumerate the objective's domain; a finite penalty is not a hard filter."""
    rows = tuple((a, cost(scenario, a)) for a in assignments(scenario))
    minimum = min(value for _, value in rows)
    return tuple(
        (a, value) for a, value in rows if math.isclose(value, minimum, abs_tol=1e-12)
    )


def distribution(scenario: Scenario, program: synth.Program) -> dict[Assignment, float]:
    values = dict(
        program.probabilities(
            fq.run(program), registers=scenario.registers, threshold=0
        )
    )
    mean_cost(scenario, values)
    return values


def mean_cost(scenario: Scenario, probabilities: dict[Assignment, float]) -> float:
    """Exact local mean replaces the source provider's sampled cost estimate."""
    if (
        not probabilities
        or any(not math.isfinite(p) or p < 0 for p in probabilities.values())
        or abs(sum(probabilities.values()) - 1) > 1e-10
    ):
        raise ValueError("normalized finite probabilities are required")
    return sum(
        probability * cost(scenario, a) for a, probability in probabilities.items()
    )


def sample(
    scenario: Scenario, probabilities: dict[Assignment, float], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    """Draw actual joint assignments and rank by the unfiltered source objective."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    mean_cost(scenario, probabilities)
    keys = tuple(sorted(probabilities))
    draws = torch.multinomial(
        torch.tensor([probabilities[a] for a in keys], dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=len(keys))
    rows = (
        Sample(
            a,
            cost(scenario, a),
            int(counts[i]),
            probabilities[a],
            sum(a[:3]) == 6 if scenario.constraint == "equality" else sum(a[:3]) <= 6,
            scenario.penalty == 0 or sum(a) == 6,
        )
        for i, a in enumerate(keys)
        if int(counts[i]) > 0
    )
    return tuple(sorted(rows, key=lambda row: (row.cost, row.assignment)))


class _EvaluationBudgetError(Exception):
    """Stop before a callback would execute beyond the source evaluation cap."""


def run_workflow(
    scenario: Scenario, *, shots: int = 1000, seed: int = 1940
) -> Workflow:
    """Run native COBYLA with the source's actual objective-call budget.

    SciPy COBYLA maxiter counts function evaluations; native maxiter counts
    trust-region iterations. The callback cap preserves the former explicitly.
    Select only parameters actually evaluated, record nonconvergence, and keep
    final readout executions separate. No optimizer trajectory equivalence.
    """
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) >= scenario.evaluation_budget:
            raise _EvaluationBudgetError()
        values = tuple(float(v) for v in parameters)
        value = mean_cost(
            scenario, distribution(scenario, build_program(scenario, values))
        )
        trace.append(Evaluation(values, value))
        return torch.tensor(value, dtype=torch.float64)

    termination: Literal["evaluation_budget", "optimizer_returned"] = (
        "optimizer_returned"
    )
    converged = False
    try:
        result = CobylaOptimizer(maxiter=200, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(initial_parameters(scenario), dtype=torch.float64)
        )
        converged = result.converged
    except _EvaluationBudgetError:
        termination = "evaluation_budget"
    selected = min(trace, key=lambda row: row.cost)
    program = build_program(scenario, selected.parameters)
    samples = sample(scenario, distribution(scenario, program), shots=shots, seed=seed)
    return Workflow(
        scenario, program, tuple(trace), selected, samples, termination, converged
    )


def verify(scenario: Scenario, parameters: tuple[float, ...]) -> StateCheck:
    """Compare absolute complex amplitudes, including source constant phases."""
    program = build_program(scenario, parameters)
    slack_weights = (1.0, 2.0, 4.0) if scenario.constraint == "inequality" else ()
    widths = (2, 2, 2, 3) if slack_weights else (2, 2, 2)
    factors = order_register_factors(
        portfolio_factors(
            2, slack_weights, scenario.penalty, scenario.return_scale, 1.0
        ),
        widths,
    )
    expected = fq.run(
        factored_cost_circuit(scenario.width, factors, parameters)
    ).statevector()
    actual = fq.run(program).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    logical = tuple(w for name in scenario.registers for w in program.registers[name])
    width = actual.numel().bit_length() - 1
    hidden = tuple(w for w in range(width) if w not in logical)
    state = (
        actual.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(1 << scenario.width, -1)
    )
    error = float((state[:, 0] - expected.flatten()).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=3e-11, rtol=0)
    assert dirty < 1e-22
    resources = program.report["resources"]
    assert (
        isinstance(resources, ResourceEstimate) and resources.n_qubits == scenario.width
    )
    return StateCheck(parameters, error, dirty, resources)


def plot_trace(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(range(1, len(workflow.trace) + 1), [row.cost for row in workflow.trace])
    axis.set(
        xlabel="Objective evaluations",
        ylabel="Exact mean encoded cost",
        title=workflow.scenario.name,
    )
    return figure


def plot_samples(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist(
        [row.cost for row in workflow.samples],
        bins=30,
        weights=[
            row.count / sum(s.count for s in workflow.samples)
            for row in workflow.samples
        ],
    )
    axis.axvline(
        classical_optima(workflow.scenario)[0][1],
        color="red",
        label="Classical objective minimum",
    )
    axis.set(
        xlabel="Joint encoded objective",
        ylabel="Observed probability",
        title=workflow.scenario.name,
    )
    axis.legend()
    return figure
