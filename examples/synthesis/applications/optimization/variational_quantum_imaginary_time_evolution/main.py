"""Construct the complete six-generator imaginary-time ansatz natively."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

P0 = fq.Y(1) @ fq.Z(2)
P1 = fq.Z(2) @ fq.Y(3)
P2 = fq.Y(0) @ fq.Z(2)
P3 = fq.Z(1) @ fq.Y(3)
P4 = fq.Y(0) @ fq.Z(1)
P5 = fq.Y(0) @ fq.Z(3)


@qfunc
def main(nodes: Output[QArray[QBit]], angles: list[float]) -> None:
    allocate(4, nodes)
    hadamard_transform(nodes)
    suzuki_trotter(P0, angles[0], 1, 1, nodes)
    suzuki_trotter(P1, angles[1], 1, 1, nodes)
    suzuki_trotter(P2, angles[2], 1, 1, nodes)
    suzuki_trotter(P3, angles[3], 1, 1, nodes)
    suzuki_trotter(P4, angles[4], 1, 1, nodes)
    suzuki_trotter(P5, angles[5], 1, 1, nodes)


def build_program() -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"angles": [0.13, -0.21, 0.32, -0.17, 0.29, -0.41]}
    )


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("nodes",))))


if __name__ == "__main__":
    run_example()
