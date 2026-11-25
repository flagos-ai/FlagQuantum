"""Encode seven classical features as Bloch rotations on four qubits."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import RX, RZ, Output, QArray, QBit, allocate, qfunc


@qfunc
def encode_pairs(data: list[float], x: QArray[QBit]) -> None:
    for index in range(3):
        RX(math.pi * data[2 * index], x[index])
        RZ(math.pi * data[2 * index + 1], x[index])
    RX(2 * math.pi * data[6], x[3])


@qfunc
def main(data: list[float], x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    encode_pairs(data, x)


def build_program(data: Sequence[float]) -> synth.Program:
    """Preserve the original tutorial's fixed seven-feature input shape."""
    if len(data) != 7:
        raise ValueError("Bloch encoding requires exactly seven features")
    return synth.synthesize(main, classical_inputs={"data": list(data)})


if __name__ == "__main__":
    program = build_program([0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8])
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
