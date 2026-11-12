"""Run native feature encoding and a fixed-shape variational layer."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import RX, RZ, RZZ, Output, QArray, QBit, R, allocate, qfunc


@qfunc
def encode(data: list[float], q: QArray[QBit]) -> None:
    for index in range(len(data)):
        RX(math.pi * data[index], q[index])
    for index in range(len(data)):
        RZ(math.pi * data[index], q[index])


@qfunc
def ansatz(weights: list[float], q: QArray[QBit]) -> None:
    for index in range(len(q)):
        R(weights[index], math.pi / 2, q[index])
    for index in range(len(q) - 1):
        RZZ(weights[len(q) + index], q[index : index + 2])
    RZZ(weights[7], q[0:2])


@qfunc
def main(
    input_vec: list[float], weight: list[float], result: Output[QArray[QBit]]
) -> None:
    allocate(4, result)
    encode(input_vec, result)
    ansatz(weight, result)


def build_program(data: Sequence[float], weights: Sequence[float]) -> synth.Program:
    """Keep the source program's fixed four inputs and 8 weights."""
    if len(data) != 4 or len(weights) != 8:
        raise ValueError("the source shape requires four features and 8 weights")
    return synth.synthesize(
        main, classical_inputs={"input_vec": list(data), "weight": list(weights)}
    )


if __name__ == "__main__":
    program = build_program([0.2, -0.3, 0.4, -0.5], [0.1 * (i + 1) for i in range(8)])
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("result",))))
