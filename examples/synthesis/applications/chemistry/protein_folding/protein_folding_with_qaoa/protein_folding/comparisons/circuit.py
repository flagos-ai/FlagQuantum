"""Independent classical lattice directions and a complete energy-table Circuit."""

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import factored_cost_circuit
from examples.synthesis.quadratic_reference import order_register_factors

BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.003 + 0.001 * i) for i in range(10)))
SOURCE_FIELDS = ("v.f", "v.interaction")


def direction(first: int, second: int) -> tuple[int, int, int, int]:
    return (
        first * second,
        (1 - first) * (1 - second),
        first * (1 - second),
        second * (1 - first),
    )


def cost(bits: tuple[int, ...]) -> int:
    first, second, third = (direction(bits[2 * i], bits[2 * i + 1]) for i in range(3))
    short = tuple(a - b for a, b in zip(first, second, strict=True))
    long = tuple(a - b + c for a, b, c in zip(first, second, third, strict=True))
    short_contact = (
        -short[0] ** 2 - short[1] ** 2 - (short[2] + 1) ** 2 - (short[3] - 1) ** 2 + 2
    ) ** 2
    long_contact = (
        -long[0] ** 2 - long[2] ** 2 - long[1] ** 2 - (long[3] - 1) ** 2 + 2
    ) ** 2
    closure = (
        long[0] ** 2 + long[1] ** 2 + (long[2] + 1) ** 2 + (long[3] - 1) ** 2 - 1
    ) ** 2
    overlaps = (1 - bits[0] ** 2) * (1 - (1 - bits[1]) ** 2)
    overlaps += sum(
        (1 - (bits[i] - bits[i + 2]) ** 2) * (1 - (bits[i + 1] - bits[i + 3]) ** 2)
        for i in (0, 2)
    )
    return (
        -bits[6] * (300 * short_contact + 300 * long_contact + 500 * closure - 2600)
        - 500 * overlaps
    )


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    # Table order is physical-layout independent: six folding bits then activity.
    table = tuple(
        float(cost(tuple((basis >> bit) & 1 for bit in range(7))))
        for basis in range(128)
    )
    factors = order_register_factors([(tuple(range(7)), table)], (6, 1))
    alternating = tuple(value for i in range(5) for value in (params[i], params[5 + i]))
    return factored_cost_circuit(7, factors, alternating)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params), reference_circuit(params), ("fold", "active")
        )


if __name__ == "__main__":
    run_example()
