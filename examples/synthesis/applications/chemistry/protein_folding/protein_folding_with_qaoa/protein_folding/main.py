"""Retain the lattice-folding contact energies in named quantum helpers."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    H,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def short_contact(
    gamma: float, first: QArray[QBit], second: QArray[QBit], active: QBit
) -> None:
    phase(
        -300
        * active
        * (
            -((first[0] * first[1] - second[0] * second[1]) ** 2)
            - ((1 - first[0]) * (1 - first[1]) - (1 - second[0]) * (1 - second[1])) ** 2
            - (first[0] * (1 - first[1]) - second[0] * (1 - second[1]) + 1) ** 2
            - (first[1] * (1 - first[0]) - second[1] * (1 - second[0]) - 1) ** 2
            + 2
        )
        ** 2,
        gamma,
    )


@qfunc
def long_contact(
    gamma: float,
    first: QArray[QBit],
    second: QArray[QBit],
    third: QArray[QBit],
    active: QBit,
) -> None:
    phase(
        -300
        * active
        * (
            -((first[0] * first[1] - second[0] * second[1] + third[0] * third[1]) ** 2)
            - (
                first[0] * (1 - first[1])
                - second[0] * (1 - second[1])
                + third[0] * (1 - third[1])
            )
            ** 2
            - (
                (1 - first[0]) * (1 - first[1])
                - (1 - second[0]) * (1 - second[1])
                + (1 - third[0]) * (1 - third[1])
            )
            ** 2
            - (
                first[1] * (1 - first[0])
                - second[1] * (1 - second[0])
                + third[1] * (1 - third[0])
                - 1
            )
            ** 2
            + 2
        )
        ** 2,
        gamma,
    )
    phase(
        -500
        * active
        * (
            (first[0] * first[1] - second[0] * second[1] + third[0] * third[1]) ** 2
            + (
                (1 - first[0]) * (1 - first[1])
                - (1 - second[0]) * (1 - second[1])
                + (1 - third[0]) * (1 - third[1])
            )
            ** 2
            + (
                first[0] * (1 - first[1])
                - second[0] * (1 - second[1])
                + third[0] * (1 - third[1])
                + 1
            )
            ** 2
            + (
                first[1] * (1 - first[0])
                - second[1] * (1 - second[0])
                + third[1] * (1 - third[0])
                - 1
            )
            ** 2
            - 1
        )
        ** 2,
        gamma,
    )
    phase(2600 * active, gamma)


@qfunc
def overlap(gamma: float, first: QArray[QBit], second: QArray[QBit]) -> None:
    phase(
        -500 * (1 - (first[0] - second[0]) ** 2) * (1 - (first[1] - second[1]) ** 2),
        gamma,
    )


@qfunc
def main(params: list[float], fold: Output[QArray[QBit]], active: Output[QBit]) -> None:
    allocate(6, fold)
    allocate(active)
    hadamard_transform(fold)
    H(active)
    for i in range(5):
        short_contact(params[i], fold[0:2], fold[2:4], active)
        long_contact(params[i], fold[0:2], fold[2:4], fold[4:6], active)
        phase(-500 * (1 - fold[0] ** 2) * (1 - (1 - fold[1]) ** 2), params[i])
        overlap(params[i], fold[0:2], fold[2:4])
        overlap(params[i], fold[2:4], fold[4:6])
        for bit in range(6):
            RX(params[5 + i], fold[bit])
        RX(params[5 + i], active)


DEFAULT_PARAMS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 10:
        raise ValueError("Five folding layers require ten real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(program.report["resources"])
    print(dict(program.probabilities(fq.run(program), registers=("fold", "active"))))


if __name__ == "__main__":
    run_example()
