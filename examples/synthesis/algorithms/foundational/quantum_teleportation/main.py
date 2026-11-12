"""Deferred-measurement teleportation with private retained outcome qubits."""

from math import pi

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    H,
    Output,
    QBit,
    R,
    allocate,
    control,
    drop,
    phase,
    qfunc,
)


@qfunc
def send(alice: QBit, bell: QBit, bob: QBit) -> None:
    CX(alice, bell)
    H(alice)
    CX(bell, bob)
    control(alice, lambda: phase(bob, pi))


@qfunc
def main(theta: float, phi: float, bob_qubit: Output[QBit]) -> None:
    alice: QBit = QBit()
    bell: QBit = QBit()
    allocate(bob_qubit)
    allocate(alice)
    allocate(bell)
    R(theta, pi / 2, alice)
    phase(alice, phi)
    H(bell)
    CX(bell, bob_qubit)
    send(alice, bell, bob_qubit)
    drop(alice)
    drop(bell)


def build_program(theta: float = pi / 2, phi: float = 0.0) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={"theta": theta, "phi": phi},
        max_qubits=4,
        max_workspace=1,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("bob_qubit",))))
    print(program.report["resources"])
