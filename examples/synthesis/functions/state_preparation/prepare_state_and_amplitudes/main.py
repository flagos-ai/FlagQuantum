"""Prepare host-generated probability and signed-amplitude vectors natively."""

from collections import Counter
from math import fsum, sqrt
from random import Random

import flagquantum as fq
from flagquantum.synth import (
    Output,
    QNum,
    prepare_amplitudes,
    prepare_state,
    qfunc,
    synthesize,
)


@qfunc
def distribution(probabilities: list[float], x: Output[QNum]) -> None:
    prepare_state(probabilities, 0.01, x)


@qfunc
def signed_state(amplitudes: list[float], x: Output[QNum]) -> None:
    prepare_amplitudes(amplitudes, 0, x)


def run_example() -> None:
    probabilities = [0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06]
    program = synthesize(
        distribution, classical_inputs={"probabilities": probabilities}
    )
    result = fq.run(program)
    named = dict(program.probabilities(result, registers=("x",)))
    samples = Random(17).choices(list(named), weights=list(named.values()), k=1000)
    counts = Counter(samples)
    print("Sampled probabilities:", [counts.get((i,), 0) / 1000 for i in range(8)])

    amplitudes = [-1 + 2 * i / 7 for i in range(8)]
    norm = sqrt(fsum(value * value for value in amplitudes))
    amplitudes = [value / norm for value in amplitudes]
    program = synthesize(
        signed_state, classical_inputs={"amplitudes": amplitudes}, max_workspace=0
    )
    state = fq.run(program).statevector().flatten()
    print("Raw amplitudes:", state.tolist())
    # There is one output register and no hidden storage. Match the source's
    # presentation convention only after retaining the raw signed state.
    global_phase = state[0] / state[0].abs()
    print("Amplitudes up to global phase:", (state / global_phase).real.tolist())


if __name__ == "__main__":
    run_example()
