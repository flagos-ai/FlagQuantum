"""Compose all four MaxCut cost/mixer layers in one native quantum program."""

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
def mixer_layer(beta: float, qba: QArray[QBit]) -> None:
    for i in range(len(qba)):
        RX(beta, qba[i])


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(5, v)
    hadamard_transform(v)
    for i in range(4):
        phase(
            -(
                v[0] * (1 - v[1])
                + v[1] * (1 - v[0])
                + v[0] * (1 - v[2])
                + v[2] * (1 - v[0])
                + v[1] * (1 - v[2])
                + v[2] * (1 - v[1])
                + v[1] * (1 - v[3])
                + v[3] * (1 - v[1])
                + v[2] * (1 - v[4])
                + v[4] * (1 - v[2])
                + v[3] * (1 - v[4])
                + v[4] * (1 - v[3])
            ),
            params[2 * i],
        )
        mixer_layer(params[2 * i + 1], v)


def build_program(
    params: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
) -> synth.Program:
    if len(params) != 8:
        raise ValueError("Four-layer MaxCut requires exactly eight real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))
