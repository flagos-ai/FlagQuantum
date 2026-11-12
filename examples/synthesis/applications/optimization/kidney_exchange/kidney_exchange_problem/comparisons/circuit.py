"""Independent classical assignment costs, truth factors and Circuit mixers."""

import flagquantum as fq
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
)
from examples.synthesis.quadratic_reference import order_register_factors

WEIGHTS = (0.9, 0.7, 0.6, 0.8, 0.75, 0.65, 0.85, 0.8, 0.7)


def cost(bits: tuple[int, ...]) -> float:
    utility = sum(weight * bit for weight, bit in zip(WEIGHTS, bits, strict=True))
    conflicts = sum(
        bits[first] * bits[second]
        for first in range(9)
        for second in range(first + 1, 9)
        if first // 3 == second // 3 or first % 3 == second % 3
    )
    return utility - 2 * conflicts


def cost_factors() -> list[CostFactor]:
    factors: list[CostFactor] = [
        ((bit,), (0.0, weight)) for bit, weight in enumerate(WEIGHTS)
    ]
    for first in range(9):
        for second in range(first + 1, 9):
            if first // 3 == second // 3 or first % 3 == second % 3:
                factors.append(((first, second), (0.0, 0.0, 0.0, -2.0)))
    return factors


BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(10)))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(5) for value in (params[i], params[5 + i]))
    return factored_cost_circuit(9, cost_factors(), interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("matches",))


SOURCE_FIELDS = (
    "v.x_donor1_patient1",
    "v.x_donor1_patient2",
    "v.x_donor1_patient3",
    "v.x_donor2_patient1",
    "v.x_donor2_patient2",
    "v.x_donor2_patient3",
    "v.x_donor3_patient1",
    "v.x_donor3_patient2",
    "v.x_donor3_patient3",
)


def source_reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(5) for value in (params[i], params[5 + i]))
    return factored_cost_circuit(
        9,
        order_register_factors(cost_factors(), (1, 1, 1, 1, 1, 1, 1, 1, 1)),
        interleaved,
    )


if __name__ == "__main__":
    run_example()
