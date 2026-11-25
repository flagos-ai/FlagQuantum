"""Run both original phase estimation stages with checked native callbacks."""

import math
from collections import Counter
from random import Random
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    X,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    qpe,
    suzuki_trotter,
)


@qfunc
def controlled_z(state: QArray[QBit]) -> None:
    # CRZ(pi): preserve both -i and +i branches of the controlled rotation.
    phase(state[0] * (2 * state[1] - 1), math.pi / 2)


@qfunc
def first(
    state: Output[QArray[QBit]], phase_: Output[Annotated[QNum, (2, False, 2)]]
) -> None:
    allocate(phase_)
    allocate(2, state)
    X(state[0])
    X(state[1])
    qpe(lambda: controlled_z(state), phase_)


# Source axes [I,Z] act on state[1]; [Z,I] act on state[0].
HAMILTONIAN = -0.125 * fq.Z(1) - 0.25 * fq.Z(0) + 0.375 * fq.I(0)


@qfunc
def second(
    state: Output[QArray[QBit]], phase_: Output[Annotated[QNum, (2, False, 2)]]
) -> None:
    allocate(2, state)
    allocate(phase_)
    hadamard_transform(state)
    # All terms commute: powers of this one-step callback equal the original
    # qpe_flexible evolution at -2*pi*power, including the identity phase.
    qpe(lambda: suzuki_trotter(HAMILTONIAN, -2 * math.pi, 1, 1, state), phase_)


def run_example() -> None:
    first_program = synth.synthesize(first, max_workspace=0)
    first_result = fq.run(first_program)
    first_probabilities = dict(
        first_program.probabilities(first_result, registers=("state", "phase_"))
    )
    print("Joint state/counter:", first_probabilities)
    print(
        "Fractional phase:",
        dict(first_program.numeric_probabilities(first_result, registers=("phase_",))),
    )
    # Source leaves shots unspecified. Choose and label 1000 local demonstration shots.
    first_counts = Counter(
        Random(17).choices(
            list(first_probabilities),
            weights=list(first_probabilities.values()),
            k=1000,
        )
    )
    print("1000 seeded local samples:", first_counts)
    print(first_program.report["resources"])

    second_program = synth.synthesize(second, max_workspace=0)
    second_result = fq.run(second_program)
    second_probabilities = dict(
        second_program.probabilities(second_result, registers=("state", "phase_"))
    )
    print("Joint state/counter:", second_probabilities)
    print(
        "Fractional phase:",
        dict(
            second_program.numeric_probabilities(second_result, registers=("phase_",))
        ),
    )
    # Source leaves shots unspecified. Choose and label 1000 local demonstration shots.
    second_counts = Counter(
        Random(17).choices(
            list(second_probabilities),
            weights=list(second_probabilities.values()),
            k=1000,
        )
    )
    print("1000 seeded local samples:", second_counts)
    print(second_program.report["resources"])


if __name__ == "__main__":
    run_example()
