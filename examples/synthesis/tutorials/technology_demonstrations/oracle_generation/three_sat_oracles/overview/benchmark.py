"""Regenerate every original problem size and measure native oracle construction."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from flagquantum.compiler import estimate_resources
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.core.ir import CircuitIR
from flagquantum.errors import PlanningError

from .oracle import Formula, build_program, clause_circuit

SEED = 128
# Values published by the source for its disabled Qiskit timing loop.
HISTORICAL_QISKIT = (
    0.2850170135498047,
    2.6256730556488037,
    0.75693678855896,
    5.783859968185425,
    3.3723957538604736,
    3.9280269145965576,
    39.92809295654297,
    60.67643904685974,
    16.551968097686768,
    31.536834955215454,
    31.086618900299072,
    794.9081449508667,
)


@dataclass(frozen=True)
class Row:
    variables: int
    route: str
    seconds: float
    resources: ResourceEstimate


def sizes() -> tuple[int, ...]:
    upper = np.logspace(np.log2(24), np.log2(68), 11, base=2)
    return (*range(11, 23), *(int(np.round(value)) for value in upper))


def generate(width: int, rng: np.random.RandomState) -> Formula:
    """Keep the source draw order, pairwise-distinct rule and rejection cap."""
    direct = np.arange(width)
    for _ in range(1000):
        left, right = rng.permutation(width), rng.permutation(width)
        if np.all(left != direct) and np.all(right != direct) and np.all(left != right):
            return Formula(tuple(map(int, left)), tuple(map(int, right)))
    raise RuntimeError(f"No valid formula found within 1000 attempts for width {width}")


def verify_basis(circuit: CircuitIR, formula: Formula, word: int) -> None:
    """Apply the emitted reversible instructions to a basis word without dense storage."""
    bits = [int(bool(word & (1 << bit))) for bit in reversed(range(formula.width))]
    bits.extend([0] * (circuit.n_wires - formula.width))
    expected = bits.copy()
    sign = 1
    for operation in circuit.instructions:
        wires = operation.wires
        if operation.name == "x":
            bits[wires[0]] ^= 1
        elif operation.name in {"cx", "ccx"}:
            bits[wires[-1]] ^= int(all(bits[wire] for wire in wires[:-1]))
        elif operation.name == "z":
            sign *= 1 - 2 * bits[wires[0]]
        elif operation.name != "i":
            raise AssertionError(f"Unexpected reversible-check gate {operation.name}")
    if bits != expected or sign != (-1 if formula.satisfies(word) else 1):
        raise AssertionError(f"Phase or workspace disagrees on basis word {word}")


def build(formula: Formula) -> tuple[str, CircuitIR, ResourceEstimate]:
    """Prefer qfunc when the exact planner admits the generated benchmark circuit."""
    if formula.width >= 14:
        try:
            program = build_program(formula)
        except PlanningError:
            # This sweep uses a fixed low-memory budget for exact basis execution.
            # Direct reversible construction remains a compile-only route.
            pass
        else:
            resources = program.report["resources"]
            if not isinstance(resources, ResourceEstimate):
                raise TypeError("The synthesized oracle did not report resources")
            return "Python qfunc", program.circuit, resources
    circuit = clause_circuit(formula).to_ir()
    return "Clause Circuit", circuit, estimate_resources(circuit)


def run() -> tuple[Row, ...]:
    rng = np.random.RandomState(SEED)
    rows = []
    for width in sizes():
        formula = generate(width, rng)
        start = perf_counter()
        route, circuit, resources = build(formula)
        seconds = perf_counter() - start
        for word in (0, 1 << formula.negative[0], (1 << width) - 1):
            verify_basis(circuit, formula, word)
        rows.append(Row(width, route, seconds, resources))
    return tuple(rows)
