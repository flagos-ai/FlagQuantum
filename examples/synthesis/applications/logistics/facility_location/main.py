"""Retain the original assignment costs and every QAOA layer in native qfunc."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def cost(gamma: float, weights: list[float], assignments: QArray[QBit]) -> None:
    for facility in range(3):
        for customer in range(4):
            phase(
                -weights[4 * facility + customer]
                * assignments[4 * facility + customer],
                gamma,
            )
    for customer in range(4):
        phase(
            -10
            * (
                assignments[customer]
                + assignments[4 + customer]
                + assignments[8 + customer]
                - 1
            )
            ** 2,
            gamma,
        )
    # Count empty facilities; retain the source's squared residual and constant.
    phase(
        -10
        * (
            (1 - assignments[0])
            * (1 - assignments[1])
            * (1 - assignments[2])
            * (1 - assignments[3])
            + (1 - assignments[4])
            * (1 - assignments[5])
            * (1 - assignments[6])
            * (1 - assignments[7])
            + (1 - assignments[8])
            * (1 - assignments[9])
            * (1 - assignments[10])
            * (1 - assignments[11])
            - 1
        )
        ** 2,
        gamma,
    )


@qfunc
def main(params: list[float], assignments: Output[QArray[QBit]]) -> None:
    allocate(12, assignments)
    hadamard_transform(assignments)
    for layer in range(5):
        cost(
            params[layer],
            [0.02, 0.14, 0.62, 0.11, 0.99, 0.22, 0.91, 0.09, 0.4, 0.76, 0.95, 0.61],
            assignments,
        )
        for bit in range(12):
            RX(params[5 + layer], assignments[bit])


DEFAULT_PARAMS = (
    0.003,
    -0.006,
    0.009000000000000001,
    -0.012,
    0.015,
    -0.018000000000000002,
    0.021,
    -0.024,
    0.027,
    -0.03,
)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 10:
        raise ValueError("Use 10 real parameters for all 5 retained layers")
    return synth.synthesize(main, classical_inputs={"params": params})


def run_example() -> None:
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())


if __name__ == "__main__":
    run_example()
