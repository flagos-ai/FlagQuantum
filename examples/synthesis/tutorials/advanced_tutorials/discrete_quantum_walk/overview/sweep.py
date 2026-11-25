"""Synthesize native arithmetic and explicit cascade candidates for CX comparison."""

from dataclasses import dataclass
from typing import Annotated, Literal

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.comparisons.circuit import (
    reference_circuit as circle_reference,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
    exact_rules,
)
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.core.ir import CircuitIR
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    control,
    qfunc,
)


@qfunc
def increment(x: QNum) -> None:
    x += 1


@qfunc
def decrement(x: QNum) -> None:
    x -= 1


@qfunc
def cascade(x: QNum) -> None:
    for order in range(9):
        # The compiler expands each callback in its current static loop binding.
        control(x[: 9 - order], lambda: X(x[9 - order]))  # noqa: B023
    X(x[0])


@qfunc
def inverse_cascade(x: QNum) -> None:
    with adjoint():
        cascade(x)


@qfunc
def arithmetic_root(x: Output[Annotated[QNum, 10]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    H(coin)
    control(coin == 0, lambda: increment(x))
    control(coin == 1, lambda: decrement(x))


@qfunc
def cascade_root(x: Output[Annotated[QNum, 10]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    H(coin)
    control(coin == 0, lambda: cascade(x))
    control(coin == 1, lambda: inverse_cascade(x))


@dataclass(frozen=True)
class Candidate:
    objective: Literal["width", "depth"]
    resources: ResourceEstimate
    circuit: CircuitIR


@dataclass(frozen=True)
class Point:
    width_limit: int
    arithmetic: Candidate
    cascade: Candidate


def cost(candidate: Candidate) -> tuple[int, int, int, str]:
    resources = candidate.resources
    return (
        resources.operation_counts.get("cx", 0),
        resources.depth,
        resources.n_qubits,
        candidate.objective,
    )


def select(
    style: Literal["arithmetic", "cascade"],
    width: int,
    conversions: dict[str, tuple[CircuitIR, ResourceEstimate]],
) -> Candidate:
    """Select the fewest actual CX gates among both requested compiler objectives."""
    if not 12 <= width <= 20:
        raise ValueError("The source sweep admits physical widths 12 through 20")
    root = arithmetic_root if style == "arithmetic" else cascade_root
    candidates = []
    objectives: tuple[Literal["width", "depth"], ...] = ("width", "depth")
    for objective in objectives:
        program = synth.synthesize(
            root,
            max_qubits=width,
            max_workspace=width - 11,
            objective=objective,
            max_operations=100_000,
            memory_limit_bytes=64 * 1024**2,
        )
        key = program.circuit.content_hash
        if key not in conversions:
            target = convert_basis(
                program.circuit,
                gates=("cx", "u3"),
                rules=exact_rules(),
                max_added_operations=100_000,
            ).program
            resources = estimate_resources(target)
            if set(resources.operation_counts) - {"cx", "u3"}:
                raise AssertionError("Converted walk escaped the named CX/U3 basis")
            conversions[key] = (target, resources)
        circuit, measure = conversions[key]
        if measure.n_qubits > width:
            raise AssertionError("The converted walk exceeded its physical width")
        candidates.append(Candidate(objective, measure, circuit))
    return min(candidates, key=cost)


def sweep() -> tuple[Point, ...]:
    conversions: dict[str, tuple[CircuitIR, ResourceEstimate]] = {}
    rows: list[Point] = []
    for width in range(12, 21):
        arithmetic = select("arithmetic", width, conversions)
        cascade = select("cascade", width, conversions)
        if rows:
            arithmetic = min(arithmetic, rows[-1].arithmetic, key=cost)
            cascade = min(cascade, rows[-1].cascade, key=cost)
        rows.append(Point(width, arithmetic, cascade))
    return tuple(rows)


def validate(candidate: Candidate) -> float:
    """Compare every physical amplitude for the original one-step input."""
    reference = fq.run(circle_reference(width=10, steps=1)).statevector().flatten()
    actual = fq.run(candidate.circuit).statevector().reshape(reference.numel(), -1)
    error = float((actual[:, 0] - reference).abs().max())
    work = float(actual[:, 1:].abs().square().sum())
    if error >= 1e-10 or work >= 1e-20:
        raise AssertionError(
            f"One-step sweep reference error {error}, workspace probability {work}"
        )
    return error
