"""Independent classical assignment costs, truth factors and Circuit mixers."""

import flagquantum as fq
from examples.synthesis.applications.logistics.facility_location.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)
from examples.synthesis.quadratic_reference import order_register_factors

WEIGHTS = (0.02, 0.14, 0.62, 0.11, 0.99, 0.22, 0.91, 0.09, 0.4, 0.76, 0.95, 0.61)


def cost(bits: tuple[int, ...]) -> float:
    delivery = sum(weight * bit for weight, bit in zip(WEIGHTS, bits, strict=True))
    unassigned = sum(
        (sum(bits[4 * facility + customer] for facility in range(3)) - 1) ** 2
        for customer in range(4)
    )
    empty = sum(not any(bits[4 * facility : 4 * facility + 4]) for facility in range(3))
    return -delivery - 10 * unassigned - 10 * (empty - 1) ** 2


def cost_factors() -> list[CostFactor]:
    factors: list[CostFactor] = [
        ((bit,), (0.0, -weight)) for bit, weight in enumerate(WEIGHTS)
    ]
    for customer in range(4):
        factors.append(
            squared_residual(
                tuple(4 * facility + customer for facility in range(3)),
                (1.0,) * 3,
                -1.0,
                -10.0,
            )
        )
    # (sum(empty)-1)^2 = 1-sum(empty)+2*sum(pairwise empty products).
    factors.append(((), (-10.0,)))
    for first in range(3):
        support = tuple(4 * first + bit for bit in range(4))
        factors.append((support, (10.0,) + (0.0,) * 15))
        for second in range(first + 1, 3):
            pair = (*support, *(4 * second + bit for bit in range(4)))
            factors.append((pair, (-20.0,) + (0.0,) * 255))
    return factors


BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(10)))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(5) for value in (params[i], params[5 + i]))
    return factored_cost_circuit(12, cost_factors(), interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params), reference_circuit(params), ("assignments",)
        )


SOURCE_FIELDS = ("v.x[0]", "v.x[1]", "v.x[2]")


def source_reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(5) for value in (params[i], params[5 + i]))
    return factored_cost_circuit(
        12, order_register_factors(cost_factors(), (4, 4, 4)), interleaved
    )


if __name__ == "__main__":
    run_example()
