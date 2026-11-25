"""Independent Circuit uses frozen cost coefficients and explicit parity ladders."""

import flagquantum as fq
from examples.synthesis.applications.optimization.qaoa_in_qaoa.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import factored_cost_circuit

TERMS: tuple[tuple[tuple[int, ...], float], ...] = (
    ((), -5.5),
    ((2, 3), 0.5),
    ((0, 3), 0.5),
    ((3, 5), 0.5),
    ((3, 4), 0.5),
    ((1, 2), 0.5),
    ((2, 4), 0.5),
    ((0, 1), 0.5),
    ((1, 5), 0.5),
    ((1, 4), 0.5),
    ((0, 5), 0.5),
    ((0, 4), 0.5),
)
WIDTH = 6
BINDINGS = (DEFAULT_PARAMS, (-0.019, 0.046, -0.057, 0.092))


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
