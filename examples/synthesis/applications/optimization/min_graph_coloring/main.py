"""Prepare and evolve the retained five-vertex, three-color cost model."""

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
def cost(gamma: float, colors: QArray[QBit]) -> None:
    phase(-3, gamma)
    for color in range(3):
        phase(
            (1 - colors[5 * color])
            * (1 - colors[5 * color + 1])
            * (1 - colors[5 * color + 2])
            * (1 - colors[5 * color + 3])
            * (1 - colors[5 * color + 4]),
            gamma,
        )
    for vertex in range(5):
        phase(
            -10 * (colors[vertex] + colors[5 + vertex] + colors[10 + vertex] - 1) ** 2,
            gamma,
        )
    # The retained source counts each undirected edge twice before squaring.
    phase(
        -40
        * (
            colors[0] * colors[4]
            + colors[0] * colors[1]
            + colors[0] * colors[3]
            + colors[1] * colors[3]
            + colors[2] * colors[3]
            + colors[5] * colors[9]
            + colors[5] * colors[6]
            + colors[5] * colors[8]
            + colors[6] * colors[8]
            + colors[7] * colors[8]
            + colors[10] * colors[14]
            + colors[10] * colors[11]
            + colors[10] * colors[13]
            + colors[11] * colors[13]
            + colors[12] * colors[13]
        )
        ** 2,
        gamma,
    )


@qfunc
def main(params: list[float], colors: Output[QArray[QBit]]) -> None:
    allocate(15, colors)
    hadamard_transform(colors)
    for layer in range(6):
        cost(params[layer], colors)
        for bit in range(15):
            RX(params[6 + layer], colors[bit])


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
)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 12:
        raise ValueError("Use 12 real parameters for all 6 retained layers")
    return synth.synthesize(main, classical_inputs={"params": params})


def run_example() -> None:
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())


if __name__ == "__main__":
    run_example()
