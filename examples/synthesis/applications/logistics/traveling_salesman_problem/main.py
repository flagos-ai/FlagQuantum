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
def cost(gamma: float, distances: list[float], visits: QArray[QBit]) -> None:
    # The retained source has three transitions, without a closing tour edge.
    for city in range(4):
        for next_city in range(4):
            for slot in range(3):
                phase(
                    -distances[4 * city + next_city]
                    * visits[4 * city + slot]
                    * visits[4 * next_city + slot + 1],
                    gamma,
                )
        phase(
            -(
                (
                    visits[4 * city]
                    + visits[4 * city + 1]
                    + visits[4 * city + 2]
                    + visits[4 * city + 3]
                    - 1
                )
                ** 2
            ),
            gamma,
        )
    for slot in range(4):
        phase(
            -(
                (
                    visits[slot]
                    + visits[4 + slot]
                    + visits[8 + slot]
                    + visits[12 + slot]
                    - 1
                )
                ** 2
            ),
            gamma,
        )


@qfunc
def main(params: list[float], visits: Output[QArray[QBit]]) -> None:
    allocate(16, visits)
    hadamard_transform(visits)
    for layer in range(8):
        cost(params[layer], [5, 1, 2, 5, 5, 5, 1, 2, 5, 1, 5, 1, 5, 5, 5, 5], visits)
        for bit in range(16):
            RX(params[8 + layer], visits[bit])


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
    0.033,
    -0.036000000000000004,
    0.039,
    -0.042,
    0.045,
    -0.048,
)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 16:
        raise ValueError("Use 16 real parameters for all 8 retained layers")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())


if __name__ == "__main__":
    run_example()
