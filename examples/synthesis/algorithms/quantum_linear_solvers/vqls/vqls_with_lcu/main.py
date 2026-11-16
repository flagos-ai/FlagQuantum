"""Compose the retained variational ansatz from native quantum helpers."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CX, H, Output, QArray, QBit, R, allocate, qfunc

DEFAULT_PARAMS = (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18)


@qfunc
def rotate_layer(params: list[float], q: QArray[QBit]) -> None:
    for index in range(len(q)):
        R(params[index], math.pi / 2, q[index])


@qfunc
def cz(control: QBit, target: QBit) -> None:
    H(target)
    CX(control, target)
    H(target)


@qfunc
def ansatz(params: list[float], q: QArray[QBit]) -> None:
    rotate_layer([params[0], params[1], params[2]], q)
    cz(q[0], q[1])
    cz(q[0], q[2])
    cz(q[1], q[2])
    rotate_layer([params[3], params[4], params[5]], q)
    cz(q[2], q[0])
    cz(q[2], q[1])
    cz(q[1], q[0])
    rotate_layer([params[6], params[7], params[8]], q)


@qfunc
def main(params: list[float], system_qubits: Output[QArray[QBit]]) -> None:
    allocate(3, system_qubits)
    ansatz(params, system_qubits)


def build_program(params: Sequence[float] = DEFAULT_PARAMS) -> synth.Program:
    """Bind exactly 9 angles in the retained source order."""
    if len(params) != 9:
        raise ValueError("the retained ansatz requires 9 angles")
    return synth.synthesize(
        main, classical_inputs={"params": list(params)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("system_qubits",))))
