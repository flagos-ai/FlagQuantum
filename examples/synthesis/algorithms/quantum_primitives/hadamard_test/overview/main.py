"""Estimate a controlled Fourier expectation while retaining discarded data."""

from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    allocate,
    apply,
    control,
    drop,
    qft,
    qfunc,
    within,
)


@qfunc
def controlled_qft(expectation_value: QBit, psi: QArray[QBit]) -> None:
    control(expectation_value, lambda: qft(psi))


@qfunc
def preparation_and_application(expectation_value: QBit, psi: QArray[QBit]) -> None:
    with within():
        H(expectation_value)
    with apply():
        controlled_qft(expectation_value, psi)


@qfunc
def main(expectation_value: Output[QBit]) -> None:
    psi: QArray[QBit] = QArray()
    allocate(expectation_value)
    allocate(4, psi)
    preparation_and_application(expectation_value, psi)
    drop(psi)


def run_example() -> None:
    program = synth.synthesize(main, max_workspace=0)
    result = fq.run(program)
    probabilities = dict(
        program.probabilities(result, registers=("expectation_value",))
    )
    num_shots = 100_000
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=num_shots
        )
    )
    p_0 = counts[(0,)] / num_shots
    p_1 = counts[(1,)] / num_shots
    expectation_value = 2 * p_0 - 1
    print("p_0:", p_0, "p_1:", p_1)
    print("Re(<U>):", expectation_value)
    print("Exact Re(<U>):", 0.25)
    assert abs(expectation_value - 0.25) < 0.02
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
