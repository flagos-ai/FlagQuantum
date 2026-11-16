"""Quantized piecewise-linear tanh with native coherent conditional arithmetic."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc

SLOPES = (1.0, 0.9375, 0.8125, 0.625)
INTERCEPTS = (0.0, 0.03125, 0.0625, 0.21875)


@qfunc
def piecewise(
    slopes: list[float], intercepts: list[float], source: QNum, target: QNum
) -> None:
    target ^= (
        slopes[0] * source + intercepts[0]
        if source < 0.25
        else (
            slopes[1] * source + intercepts[1]
            if source < 0.5
            else (
                slopes[2] * source + intercepts[2]
                if source < 0.75
                else slopes[3] * source + intercepts[3]
            )
        )
    )


@qfunc
def main(
    slopes: list[float],
    intercepts: list[float],
    initial: int,
    repeats: int,
    x: Output[Annotated[QNum, (4, False, 4)]],
    y: Output[Annotated[QNum, (4, False, 4)]],
) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(y)
    y ^= initial / 16
    for _ in range(repeats):
        piecewise(slopes, intercepts, x, y)


def build_program(
    initial: int = 0,
    repeats: int = 1,
    slopes: tuple[float, ...] = SLOPES,
    intercepts: tuple[float, ...] = INTERCEPTS,
) -> synth.Program:
    if (
        type(initial) is not int
        or not 0 <= initial < 16
        or type(repeats) is not int
        or not 1 <= repeats <= 2
    ):
        raise ValueError("Use a four-bit initial word and one or two XOR updates")
    if len(slopes) != 4 or len(intercepts) != 4:
        raise ValueError("Use four slopes and four intercepts")
    return synth.synthesize(
        main,
        classical_inputs={
            "slopes": list(slopes),
            "intercepts": list(intercepts),
            "initial": initial,
            "repeats": repeats,
        },
        max_workspace=2,
        max_qubits=10,
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x", "y"))))
    print(program.report["resources"])
