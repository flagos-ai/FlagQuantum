"""Construct the single-qubit Bloch-feature overlap with native helpers."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import RX, RZ, Output, QBit, allocate, qfunc


@qfunc
def encode_on_bloch(data: list[float], q: QBit) -> None:
    RX(math.pi * data[0], q)
    RZ(math.pi * data[1], q)


@qfunc
def undo_bloch_encoding(data: list[float], q: QBit) -> None:
    RZ(-math.pi * data[1], q)
    RX(-math.pi * data[0], q)


@qfunc
def main(data1: list[float], data2: list[float], qba: Output[QBit]) -> None:
    allocate(qba)
    encode_on_bloch(data1, qba)
    undo_bloch_encoding(data2, qba)


def build_program() -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"data1": [0.23, -0.41], "data2": [-0.17, 0.37]}
    )


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
