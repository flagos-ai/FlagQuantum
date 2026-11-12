"""Compose the retained ADAPT-QAOA cost and selected mixer as native qfuncs."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    suzuki_trotter,
)

MIXER = fq.X(0) + fq.X(1) + fq.X(2) + fq.X(3) + fq.X(4) + fq.X(5) + fq.X(6) + fq.X(7)


@qfunc
def cost_layer(gamma: float, v: QArray[QBit]) -> None:
    phase(
        2
        * (
            v[0] * v[1]
            + v[0] * v[2]
            + v[1] * v[3]
            + v[2] * v[3]
            + v[4] * v[5]
            + v[4] * v[6]
            + v[5] * v[7]
            + v[6] * v[7]
        )
        - (
            0.7525 * v[0] * v[4]
            + 0.8004 * v[0] * v[5]
            + 0.1 * v[0]
            + 0.1 * v[1]
            + 0.1924 * v[2] * v[4]
            + 0.1 * v[2]
            + 0.1 * v[4]
            + 0.1 * v[5]
            + 0.1581 * v[3] * v[6]
            + 0.9165 * v[3] * v[7]
            + 0.1 * v[3]
            + 0.1 * v[6]
            + 0.1 * v[7]
            + 3.4127
        )
        + 0.9382 * v[1] * v[6]
        + 0.2143 * v[1] * v[7]
        + 0.9182 * v[2] * v[5],
        gamma,
    )


@qfunc
def selected_mixer(beta: float, v: QArray[QBit]) -> None:
    suzuki_trotter(MIXER, beta, 1, 1, v)


@qfunc
def ansatz(gamma: float, beta: float, v: QArray[QBit]) -> None:
    cost_layer(gamma, v)
    selected_mixer(beta, v)


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(8, v)
    hadamard_transform(v)
    ansatz(params[0], params[1], v)


def build_program(params: tuple[float, ...] = (0.1, 0.2)) -> synth.Program:
    if len(params) != 2:
        raise ValueError("ADAPT-QAOA quantum program requires gamma and beta")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))
