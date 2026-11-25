"""Sample and plot both original partial-uniform state preparations."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.state_preparation.prepare_uniform_interval_state.main import (
    main as interval,
)
from examples.synthesis.functions.state_preparation.prepare_uniform_trimmed_state.main import (
    main as trimmed,
)
from flagquantum import synth


def run_example() -> None:
    import matplotlib.pyplot as plt

    for definition, size in ((trimmed, (4, 3)), (interval, (5, 3))):
        program = synth.synthesize(definition)
        result = fq.run(program)
        probabilities = dict(program.probabilities(result, registers=("x",)))
        counts = Counter(
            Random(17).choices(
                list(probabilities), weights=list(probabilities.values()), k=1000
            )
        )
        _, axes = plt.subplots(figsize=size)
        axes.bar(
            [state[0] for state in sorted(counts)],
            [count for state, count in sorted(counts.items())],
            color="skyblue",
            edgecolor="black",
        )
        axes.set_xlabel("state")
        axes.set_ylabel("shots")
        print("Probabilities:", probabilities)
        print("Counts:", dict(counts))
        print("Resources:", program.report["resources"])
    plt.show()


if __name__ == "__main__":
    run_example()
