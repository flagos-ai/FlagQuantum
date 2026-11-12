"""Encode four classical features into a native quantum register."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, R, allocate, qfunc


@qfunc
def encode_angles(data: list[float], x: QArray[QBit]) -> None:
    for index in range(len(data)):
        R(math.pi * data[index], math.pi / 2, x[index])


@qfunc
def main(data: list[float], x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    encode_angles(data, x)


def build_program(data: Sequence[float]) -> synth.Program:
    """Preserve the original tutorial's fixed four-feature input shape."""
    if len(data) != 4:
        raise ValueError("angle encoding requires exactly four features")
    return synth.synthesize(main, classical_inputs={"data": list(data)})


if __name__ == "__main__":
    program = build_program([0.2, -0.3, 0.4, 0.5])
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
