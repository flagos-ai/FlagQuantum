"""Native layered quantum program for max clique."""

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
        v[0]
        + v[1]
        + v[2]
        + v[3]
        + v[4]
        + v[5]
        + v[6]
        - 4
        * (
            v[0] * v[5]
            + v[1] * v[4]
            + v[2] * v[4]
            + v[2] * v[6]
            + v[3] * v[6]
            + v[4] * v[6]
        )
        ** 2,
        gamma,
    )


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(7, v)
    hadamard_transform(v)
    for i in range(3):
        cost_layer(params[i], v)
        for j in range(7):
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
