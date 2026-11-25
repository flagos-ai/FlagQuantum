"""Native layered quantum program for set partition."""

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
        -(
            (
                8 * v[0]
                + 18 * v[1]
                + 6 * v[2]
                + 18 * v[3]
                + 4 * v[4]
                + 12 * v[5]
                + 18 * v[6]
                + 20 * v[7]
                + 14 * v[8]
                + 16 * v[9]
                - 67
            )
            ** 2
        ),
        gamma,
    )


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    hadamard_transform(v)
    for i in range(3):
        cost_layer(params[i], v)
        for j in range(10):
            RX(params[3 + i], v[j])


def build_program(
    params: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
) -> synth.Program:
    if len(params) != 6:
        raise ValueError("This 3-layer program requires 6 real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))
