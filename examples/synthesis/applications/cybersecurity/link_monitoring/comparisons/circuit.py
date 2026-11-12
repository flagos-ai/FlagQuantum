"""Independent Circuit uses frozen cost coefficients and explicit parity ladders."""

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.link_monitoring.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import factored_cost_circuit

TERMS: tuple[tuple[tuple[int, ...], float], ...] = (
    ((), 12.5),
    ((7,), 0.5),
    ((6,), 0.5),
    ((10,), 1.0),
    ((8,), 1.0),
    ((4,), 1.0),
    ((11,), 1.5),
    ((9,), 1.5),
    ((10, 11), 0.5),
    ((9, 11), 0.5),
    ((8, 11), 0.5),
    ((7, 11), 0.5),
    ((2, 4), 0.5),
    ((1, 7), 0.5),
    ((0, 9), 0.5),
    ((9, 10), 0.5),
    ((8, 10), 0.5),
    ((5, 9), 0.5),
    ((6, 8), 0.5),
    ((4, 6), 0.5),
    ((3, 4), 0.5),
)
WIDTH = 12
BINDINGS = (DEFAULT_PARAMS, (-0.019, 0.046))


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
