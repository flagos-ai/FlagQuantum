"""Prepare and evolve the original eight-orbital Fermi-Hubbard state."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    X,
    allocate,
    phase,
    qfunc,
    suzuki_trotter,
    unitary,
)

from .physics import EXCHANGE, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9


@qfunc
def prepare_slater_det(q: QArray[QBit]) -> None:
    for bit in range(4):
        X(q[bit])
    unitary(G0, [q[4], q[3]])
    unitary(G1, [q[3], q[2]])
    unitary(G2, [q[5], q[4]])
    unitary(G3, [q[2], q[1]])
    unitary(G4, [q[6], q[5]])
    unitary(G5, [q[1], q[0]])
    unitary(G6, [q[3], q[2]])
    unitary(G7, [q[7], q[6]])
    unitary(G8, [q[2], q[1]])
    unitary(G9, [q[3], q[2]])


@qfunc
def exchange(time: float, q: QArray[QBit]) -> None:
    suzuki_trotter(EXCHANGE, time, 1, 1, q)


@qfunc
def contact(left: QBit, right: QBit) -> None:
    phase(left * right, 0.6)


@qfunc
def main(num_iter: int, qba: Output[QArray[QBit]]) -> None:
    allocate(8, qba)
    prepare_slater_det(qba)
    for step in range(num_iter):
        exchange(-0.0, qba[0:2])
        exchange(-0.0, qba[2:4])
        exchange(-0.0, qba[4:6])
        exchange(-0.0, qba[6:8])
        contact(qba[0], qba[4])
        contact(qba[2], qba[6])
        exchange(-1.570796326794897, qba[1:3])
        exchange(-1.570796326794897, qba[5:7])
        contact(qba[2], qba[6])
        contact(qba[3], qba[7])
        exchange(1.570796326794897, qba[1:3])
        exchange(1.570796326794897, qba[5:7])


def build_program(num_iter: int = 1) -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"num_iter": num_iter}, max_workspace=0
    )


def run_example() -> None:
    program = build_program(2)
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
