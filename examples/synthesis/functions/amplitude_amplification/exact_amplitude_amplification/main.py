"""Exact amplification original, existing native composition and independent Circuit.

The Qmod entry owns its ancilla. The native comparison spells the same small
algorithm through existing qfunc phase/rotation composition and returns its
explicit clean auxiliary register; it adds no native exact-amplification API."""

import math
from collections import Counter
from pathlib import Path
from random import Random

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, R, allocate, phase, qfunc

PHI = math.pi / 2


THETA = 2 * math.asin(math.sqrt(0.07))


BETA = 2 * math.acos(math.sin(math.pi / 14) / math.sqrt(0.07))


@qfunc
def main(target: Output[QBit], aux: Output[QBit]) -> None:
    allocate(target)
    allocate(aux)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    # The current native frontend uses static calls; three rounds are explicit.
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)


def sample_counts(shots: int = 1000, seed: int = 17) -> dict[tuple[int, ...], int]:
    """Sample executed named probabilities locally and check exact success."""
    if type(shots) is not int or shots < 1:
        raise ValueError("shots must be a positive integer")
    program = synth.synthesize(main)
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("target", "aux")))
    counts = dict(
        Counter(
            Random(seed).choices(
                list(probabilities), weights=list(probabilities.values()), k=shots
            )
        )
    )
    if counts != {(1, 0): shots}:
        raise AssertionError("exact amplification must succeed and clean the auxiliary")
    return counts


def trajectories() -> (
    tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]
):
    """Preserve the Notebook's standard and ancilla-tuned Grover trajectories."""
    theta = math.asin(math.sqrt(0.07))
    rounds = math.ceil(math.pi / (4 * theta) - 0.5)
    tuned = math.pi / (4 * rounds + 2)
    return tuple(
        (math.sin((2 * step + 1) * theta), math.cos((2 * step + 1) * theta))
        for step in range(rounds + 1)
    ), tuple(
        (math.sin((2 * step + 1) * tuned), math.cos((2 * step + 1) * tuned))
        for step in range(rounds + 1)
    )


def plot_trajectories(path: Path) -> None:
    """Save the source's good/bad plane comparison without a cloud dependency."""
    from matplotlib import pyplot as plt

    standard, tuned = trajectories()
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.set_aspect("equal")
    axis.axhline(0, color="gray", linewidth=1)
    axis.axvline(0, color="gray", linewidth=1)
    for states, color, label in (
        (standard, "blue", "amplitude amplification"),
        (tuned, "red", "exact amplitude amplification"),
    ):
        for step, (good, bad) in enumerate(states):
            axis.arrow(
                0,
                0,
                good,
                bad,
                head_width=0.035,
                length_includes_head=True,
                color=color,
                alpha=0.7,
            )
            axis.text(good * 1.08, bad * 1.08, str(step), color=color)
        axis.plot([], [], color=color, label=label)
    axis.set(xlim=(-1, 1.6), ylim=(-1, 1.4), xlabel="|Good>", ylabel="|Bad>")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_example(plot_path: Path | None = None) -> None:
    print(sample_counts())
    if plot_path is not None:
        plot_trajectories(plot_path)
        print(f"Saved Grover trajectories to {plot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="save the Notebook's trajectory plot")
    run_example(parser.parse_args().plot)
