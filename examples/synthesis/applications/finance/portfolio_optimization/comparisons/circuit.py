"""Independent classical covariance, encoding and Circuit phase reference."""

import flagquantum as fq
from examples.synthesis.applications.finance.portfolio_optimization.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor, factored_cost_circuit
from examples.synthesis.portfolio_reference import portfolio_cost, portfolio_factors
from examples.synthesis.quadratic_reference import order_register_factors

WIDTH = 3
SLACK_WEIGHTS = (1.0, 2.0, 3.0)
PENALTY = 10.0
RETURN_SCALE = 1.0
SIGN = -1.0
INTERLEAVED = False
BINDINGS = (DEFAULT_PARAMS, (-0.017, 0.031, 0.043, -0.029, 0.037, -0.053))


def classical_cost(bits: tuple[int, ...]) -> float:
    values = tuple(
        sum(bits[asset * WIDTH + bit] * (1 << bit) for bit in range(WIDTH))
        for asset in range(3)
    )
    slack = sum(
        coefficient * bit
        for coefficient, bit in zip(SLACK_WEIGHTS, bits[3 * WIDTH :], strict=True)
    )
    return portfolio_cost(
        (values[0], values[1], values[2]), slack, PENALTY, RETURN_SCALE, SIGN
    )


def cost_factors() -> list[CostFactor]:
    return portfolio_factors(WIDTH, SLACK_WEIGHTS, PENALTY, RETURN_SCALE, SIGN)


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    angles = (
        params
        if INTERLEAVED
        else tuple(value for i in range(3) for value in (params[i], params[3 + i]))
    )
    widths = (WIDTH, WIDTH, WIDTH) + ((3,) if SLACK_WEIGHTS else ())
    factors = order_register_factors(cost_factors(), widths)
    return factored_cost_circuit(sum(widths), factors, angles)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params),
            reference_circuit(params),
            ("a0", "a1", "a2", "slack"),
        )


if __name__ == "__main__":
    run_example()
