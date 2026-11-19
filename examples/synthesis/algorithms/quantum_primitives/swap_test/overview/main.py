"""Measure overlap with the source's full precision seed-zero amplitude data."""

import math
from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import main
from flagquantum import synth


def run_example() -> None:
    import numpy as np

    rng = np.random.RandomState(0)
    amps1 = 1 - 2 * rng.rand(8)
    amps2 = 1 - 2 * rng.rand(8)
    amps1 = amps1 / np.linalg.norm(amps1)
    amps2 = amps2 / np.linalg.norm(amps2)
    program = synth.synthesize(
        main,
        classical_inputs={"first": amps1.tolist(), "second": amps2.tolist()},
        max_workspace=0,
    )
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("test",)))
    num_shots = 100_000
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=num_shots
        )
    )
    overlap_from_swap_test = math.sqrt(2 * counts[(0,)] / num_shots - 1)
    exact_overlap = float(np.abs(amps1 @ amps2))
    print("Sampled overlap:", overlap_from_swap_test)
    print("Exact overlap:", exact_overlap)
    atol = 0.05
    assert (
        abs(overlap_from_swap_test - exact_overlap) < atol
    ), "Sampled overlap exceeds the absolute tolerance."
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
