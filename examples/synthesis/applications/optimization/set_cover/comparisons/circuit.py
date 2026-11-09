"""Independent affine constraints and bounded parity Circuit reference."""

import flagquantum as fq
from examples.synthesis.applications.optimization.set_cover.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor, factored_cost_circuit
from examples.synthesis.quadratic_reference import (
    affine_square_factors,
    order_register_factors,
)

DECISION_BITS = 8
WIDTHS = (1, 2, 2, 2, 1, 1, 2, 2, 1, 1)
OFFSETS = (8, 9, 11, 13, 15, 16, 17, 19, 21, 22)
NEIGHBORS = (
    (0, 4),
    (0, 1, 7),
    (0, 1, 5),
    (0, 1, 6),
    (1, 7),
    (2, 4),
    (2, 5, 6),
    (3, 4, 7),
    (3, 5),
    (3, 6),
)
SLACK_WEIGHTS = (
    (1.0,),
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0,),
    (1.0,),
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0,),
    (1.0,),
)
BINDINGS = (
    DEFAULT_PARAMS,
    tuple(0.005 * (i + 1) if i % 3 else -0.004 * (i + 1) for i in range(6)),
)


def residuals() -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    return [
        (
            tuple(range(offset, offset + len(weights))) + neighbors,
            weights + (-1.0,) * len(neighbors),
        )
        for offset, weights, neighbors in zip(
            OFFSETS, SLACK_WEIGHTS, NEIGHBORS, strict=True
        )
    ]


def classical_cost(bits: tuple[int, ...]) -> float:
    if len(bits) != 23:
        raise ValueError("The complete source requires 23 logical bits")
    residual_cost = sum(
        (
            1
            + sum(
                coefficient * bits[bit]
                for bit, coefficient in zip(support, coefficients, strict=True)
            )
        )
        ** 2
        for support, coefficients in residuals()
    )
    return -sum(bits[:DECISION_BITS]) - 10 * residual_cost


def cost_factors() -> list[CostFactor]:
    factors: list[CostFactor] = [((bit,), (0.0, -1.0)) for bit in range(DECISION_BITS)]
    for support, coefficients in residuals():
        factors.extend(affine_square_factors(support, coefficients, 1.0, -10.0))
    return factors


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(3) for value in (params[i], params[3 + i]))
    factors = order_register_factors(cost_factors(), (DECISION_BITS, *WIDTHS))
    return factored_cost_circuit(23, factors, interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params),
            reference_circuit(params),
            ("x", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"),
        )


if __name__ == "__main__":
    run_example()
