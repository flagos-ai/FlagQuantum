"""Native four-layer MaxCut with an explicitly selected cost helper."""

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
def maxcut_cost_layer(gamma: float, vertices: QArray[QBit]) -> None:
    phase(
        (
            vertices[0] * (1 - vertices[1])
            + vertices[1] * (1 - vertices[0])
            + vertices[0] * (1 - vertices[2])
            + vertices[2] * (1 - vertices[0])
            + vertices[1] * (1 - vertices[2])
            + vertices[2] * (1 - vertices[1])
            + vertices[1] * (1 - vertices[3])
            + vertices[3] * (1 - vertices[1])
            + vertices[2] * (1 - vertices[4])
            + vertices[4] * (1 - vertices[2])
            + vertices[3] * (1 - vertices[4])
            + vertices[4] * (1 - vertices[3])
        )
        / 6,
        gamma,
    )


@qfunc
def mixer_layer(beta: float, vertices: QArray[QBit]) -> None:
    for vertex in range(5):
        RX(beta, vertices[vertex])


@qfunc
def ansatz(params: list[float], vertices: QArray[QBit]) -> None:
    for layer in range(4):
        maxcut_cost_layer(params[layer], vertices)
        mixer_layer(params[4 + layer], vertices)


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(5, v)
    hadamard_transform(v)
    ansatz(params, v)


DEFAULT_PARAMS = (
    0.1,
    0.2,
    0.30000000000000004,
    0.4,
    0.5,
    0.6000000000000001,
    0.7000000000000001,
    0.8,
)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 8:
        raise ValueError("Four MaxCut layers require eight real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
