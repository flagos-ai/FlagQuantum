"""Native layered quantum program for ising model."""

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
def cost_layer(gamma: float, v: QArray[QBit]) -> None:
    phase(
        -20
        * (
            (2 * v[0] - 1)
            + (2 * v[1] - 1)
            + (2 * v[2] - 1)
            + (2 * v[3] - 1)
            + (2 * v[4] - 1)
            + (2 * v[5] - 1)
        )
        + 10
        * (
            (2 * v[0] - 1) * (2 * v[1] - 1)
            + (2 * v[0] - 1) * (2 * v[5] - 1)
            + (2 * v[1] - 1) * (2 * v[2] - 1)
            + (2 * v[2] - 1) * (2 * v[3] - 1)
            + (2 * v[3] - 1) * (2 * v[4] - 1)
            + (2 * v[4] - 1) * (2 * v[5] - 1)
        ),
        gamma,
    )


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(6, v)
    hadamard_transform(v)
    for i in range(5):
        cost_layer(params[i], v)
        for j in range(6):
            RX(params[5 + i], v[j])


def build_program(
    params: tuple[float, ...] = (
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
    )
) -> synth.Program:
    if len(params) != 10:
        raise ValueError("This 5-layer program requires 10 real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))
