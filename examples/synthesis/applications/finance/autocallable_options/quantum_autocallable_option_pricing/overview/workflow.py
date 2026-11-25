"""Run the local state preparation and its first Grover iterate at source scale."""

import math
from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.main import (
    main as preparation,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

from . import model, reference

MAX_QUBITS = 24
MAX_OPERATIONS = 100_000
MEMORY_LIMIT_BYTES = 1024**3


@dataclass(frozen=True)
class Check:
    iterations: int
    resources: ResourceEstimate
    probability: float
    state_error: float
    workspace_probability: float


@dataclass(frozen=True)
class Result:
    grid: model.Grid
    transform: model.Encoding
    paths: tuple[model.Path, ...]
    expectation: float
    exact_price: float
    checks: tuple[Check, ...]


def build_program(
    grid: model.Grid,
    transform: model.Encoding,
    iterations: int,
    *,
    max_qubits: int = MAX_QUBITS,
) -> synth.Program:
    """Bind the regenerated model to the original finite preparation kernel."""
    if type(iterations) is not int or not 0 <= iterations <= 4:
        raise ValueError("The demonstration admits 0 to 4 static Grover iterations")
    if grid != model.gaussian_grid() or transform != model.encoding(grid):
        raise ValueError(
            "This preparation is specialized to the source financial contract"
        )
    return synth.synthesize(
        preparation,
        classical_inputs={
            "k": iterations,
            "probabilities": list(grid.probabilities),
            "exponential": list(transform.reference_probabilities()),
            "bounds": list(model.comparisons(grid, transform)),
        },
        max_qubits=max_qubits,
        max_operations=MAX_OPERATIONS,
        memory_limit_bytes=MEMORY_LIMIT_BYTES,
    )


def compare(
    grid: model.Grid,
    transform: model.Encoding,
    expected: torch.Tensor,
    iterations: int,
) -> Check:
    """Check the entire purification, its sign and all compiler workspace."""
    program = build_program(grid, transform, iterations)
    result = fq.run(program)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("The synthesized program did not report resources")
    actual = result.statevector().reshape(expected.numel(), -1)
    error = float((actual[:, 0] - expected).abs().max())
    work = float(actual[:, 1:].abs().square().sum())
    distribution = program.probabilities(
        result, registers=("indicator",), max_amplitudes=actual.numel()
    )
    return Check(iterations, resources, dict(distribution)[(1,)], error, work)


def run() -> Result:
    """Reproduce the live pricing workflow and prepare a static Grover circuit."""
    grid = model.gaussian_grid()
    transform = model.encoding(grid)
    paths = model.paths(grid)
    expectation = math.fsum(row.probability * row.discounted_payoff for row in paths)
    prepared = reference.prepared_state(grid, transform, paths)
    checks = (
        compare(grid, transform, prepared, 0),
        compare(grid, transform, reference.grover_once(prepared), 1),
    )
    return Result(
        grid,
        transform,
        paths,
        expectation,
        transform.decode(checks[0].probability),
        checks,
    )


def print_report(result: Result) -> None:
    print(
        "Path totals:",
        sorted(Counter(row.payment.value for row in result.paths).items()),
    )
    print("Classical finite-grid payoff:", result.expectation)
    print("Decoded exact native payoff:", result.exact_price)
    for check in result.checks:
        resource = check.resources
        print(
            f"Grover k={check.iterations}:",
            resource.n_qubits,
            resource.n_operations,
            resource.depth,
            check.probability,
            check.state_error,
            check.workspace_probability,
        )
    print("Historical IQAE payoff, source only:", -3.5079217726515903)
    print(
        "Historical IQAE confidence interval, source only:",
        tuple(
            result.transform.decode(probability)
            for probability in (0.119158741, 0.1197666)
        ),
    )
