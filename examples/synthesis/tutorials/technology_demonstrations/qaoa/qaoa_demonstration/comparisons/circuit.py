"""Independent Circuit uses frozen cost coefficients and explicit parity ladders."""

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import factored_cost_circuit
from examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.main import (
    DEFAULT_PARAMS,
    build_program,
)

TERMS: tuple[tuple[tuple[int, ...], float], ...] = (
    ((), 30.5),
    ((6,), -6.5),
    ((1,), -6.5),
    ((5,), -8.5),
    ((3,), -8.5),
    ((4,), -17.5),
    ((2,), -23.5),
    ((0,), -23.5),
    ((5, 6), 1.0),
    ((3, 6), 1.0),
    ((3, 5), 1.0),
    ((1, 5), 1.0),
    ((1, 3), 1.0),
    ((4, 6), 2.0),
    ((1, 4), 2.0),
    ((0, 5), 4.0),
    ((2, 3), 4.0),
    ((0, 2), 16.0),
    ((2, 6), 3.0),
    ((0, 6), 3.0),
    ((4, 5), 3.0),
    ((3, 4), 3.0),
    ((1, 2), 3.0),
    ((0, 1), 3.0),
    ((1, 6), 7.0),
    ((2, 5), 9.0),
    ((0, 3), 9.0),
    ((2, 4), 13.0),
    ((0, 4), 13.0),
    ((2, 5, 6), -1.0),
    ((1, 5, 6), -1.0),
    ((2, 4, 6), -1.0),
    ((0, 4, 6), -1.0),
    ((1, 3, 6), -1.0),
    ((0, 3, 6), -1.0),
    ((0, 2, 6), -1.0),
    ((0, 4, 5), -1.0),
    ((2, 3, 5), -1.0),
    ((0, 3, 5), -1.0),
    ((1, 2, 5), -1.0),
    ((2, 3, 4), -1.0),
    ((1, 2, 4), -1.0),
    ((0, 1, 4), -1.0),
    ((0, 1, 3), -1.0),
    ((0, 1, 2), -1.0),
    ((1, 4, 6), -2.0),
    ((0, 2, 5), -4.0),
    ((0, 2, 3), -4.0),
    ((0, 2, 4), -8.0),
    ((1, 2, 6), -3.0),
    ((0, 1, 6), -3.0),
    ((2, 4, 5), -3.0),
    ((0, 3, 4), -3.0),
    ((1, 2, 5, 6), 1.0),
    ((1, 2, 4, 6), 1.0),
    ((0, 1, 4, 6), 1.0),
    ((0, 1, 3, 6), 1.0),
    ((0, 1, 2, 6), 1.0),
    ((0, 2, 4, 5), 1.0),
    ((0, 2, 3, 5), 1.0),
    ((0, 2, 3, 4), 1.0),
)
WIDTH = 7
BINDINGS = (
    DEFAULT_PARAMS,
    (
        -0.019,
        0.046,
        -0.057,
        0.092,
        -0.095,
        0.138,
        -0.133,
        0.184,
        -0.171,
        0.23,
        -0.209,
        0.276,
        -0.247,
        0.322,
        -0.285,
        0.368,
        -0.323,
        0.414,
        -0.361,
        0.46,
    ),
)


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    factors = [
        (
            support,
            tuple(
                -coefficient * (-1 if basis.bit_count() % 2 else 1)
                for basis in range(1 << len(support))
            ),
        )
        for support, coefficient in TERMS
    ]
    return factored_cost_circuit(WIDTH, factors, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("target",))


if __name__ == "__main__":
    run_example()
