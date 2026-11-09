"""Compose the retained variational ansatz from native quantum helpers."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CX, Output, QArray, QBit, R, X, allocate, qfunc

DEFAULT_PARAMS = (
    0.17,
    -0.22,
    0.31,
    0.48,
    -0.51,
    0.63,
    0.72,
    -0.83,
    0.96,
    0.12,
    -0.26,
    0.37,
)


@qfunc
def ansatz(params: list[float], q: QArray[QBit]) -> None:
    for layer in range(3):
        for index in range(4):
            X(q[index])
            R(params[4 * layer + index], math.pi / 2, q[index])
        for index in range(3):
            CX(q[index], q[index + 1])


@qfunc
def main(params: list[float], state: Output[QArray[QBit]]) -> None:
    allocate(4, state)
    ansatz(params, state)


def build_program(params: Sequence[float] = DEFAULT_PARAMS) -> synth.Program:
    """Bind exactly 12 angles in the retained source order."""
    if len(params) != 12:
        raise ValueError("the retained ansatz requires 12 angles")
    return synth.synthesize(
        main, classical_inputs={"params": list(params)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("state",))))
