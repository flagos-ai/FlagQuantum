"""Retain both Simon oracles, both 250/300-shot secret recoveries and plots."""

from collections import Counter
from random import Random
from typing import TYPE_CHECKING

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Const,
    Output,
    QNum,
    allocate,
    hadamard_transform,
    min,
    qfunc,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@qfunc
def representative(x: Const[QNum], res: Output[QNum]) -> None:
    res |= min(x, x ^ 6)


@qfunc
def simple_mapping(x: Output[QNum], res: Output[QNum]) -> None:
    allocate(5, x)
    hadamard_transform(x)
    representative(x, res)


@qfunc
def simple_algorithm(x: Output[QNum], res: Output[QNum]) -> None:
    allocate(5, x)
    hadamard_transform(x)
    representative(x, res)
    hadamard_transform(x)


@qfunc
def bipartite(x: Const[QNum], res: Output[QNum]) -> None:
    allocate(6, res)
    for bit in range(2):
        CX(x[bit], res[bit])
    for bit in range(3):
        CX(x[3 + bit], res[3 + bit])
        CX(x[2], res[3 + bit])


@qfunc
def bipartite_mapping(x: Output[QNum], res: Output[QNum]) -> None:
    allocate(6, x)
    hadamard_transform(x)
    bipartite(x, res)


@qfunc
def bipartite_algorithm(x: Output[QNum], res: Output[QNum]) -> None:
    allocate(6, x)
    hadamard_transform(x)
    bipartite(x, res)
    hadamard_transform(x)


def recover_secret(samples: list[int], width: int) -> int:
    """Recover the unique nonzero null vector from decoded GF(2) samples."""
    if type(width) is not int or not 1 <= width <= 6:
        raise ValueError("Use a Simon width in 1..6.")
    if any(
        type(sample) is not int or not 0 <= sample < 1 << width for sample in samples
    ):
        raise ValueError("Samples must be unsigned decoded integers fitting the width.")
    pivots: dict[int, int] = {}
    for sample in samples:
        row = sample
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    if len(pivots) != width - 1:
        raise ValueError("Need width-1 independent GF(2) rows; increase the shots.")
    free = next(bit for bit in range(width) if bit not in pivots)
    secret = 1 << free
    for pivot in sorted(pivots):
        if (pivots[pivot] & secret).bit_count() % 2:
            secret |= 1 << pivot
    return secret


def run_workflow() -> (
    tuple[dict[str, synth.Program], dict[str, Counter[tuple[int, ...]]], dict[str, int]]
):
    programs: dict[str, synth.Program] = {}
    counts: dict[str, Counter[tuple[int, ...]]] = {}
    secrets: dict[str, int] = {}
    for name, width, callback, shots in (
        ("simple_mapping", 5, simple_mapping, 1000),
        ("simple_algorithm", 5, simple_algorithm, 250),
        ("bipartite_mapping", 6, bipartite_mapping, 1000),
        ("bipartite_algorithm", 6, bipartite_algorithm, 300),
    ):
        program = synth.synthesize(
            callback,
            max_workspace=0,
            objective="depth" if name == "bipartite_mapping" else "width",
        )
        probabilities = dict(
            program.probabilities(fq.run(program), registers=("x", "res"))
        )
        counts[name] = Counter(
            Random(17).choices(
                list(probabilities), weights=list(probabilities.values()), k=shots
            )
        )
        programs[name] = program
        if name.endswith("algorithm"):
            secrets[name] = recover_secret(
                [values[0] for values in counts[name]], width
            )
            print(name, "recovered secret", secrets[name])
        print(name, "shots:", shots, "resources:", program.report["resources"])
    assert secrets == {"simple_algorithm": 6, "bipartite_algorithm": 60}
    return programs, counts, secrets


def plot_mappings(counts: dict[str, Counter[tuple[int, ...]]]) -> list["Figure"]:
    import matplotlib.pyplot as plt

    figures: list[Figure] = []
    for name in ("simple_mapping", "bipartite_mapping"):
        sampled = {x: res for x, res in counts[name]}
        figure, axes = plt.subplots()
        axes.plot(list(sampled), list(sampled.values()), "o")
        axes.minorticks_on()
        axes.grid(axis="both", which="both")
        axes.set_xlabel("$x$")
        axes.set_ylabel("$f(x)$")
        figures.append(figure)
        plt.show()
    return figures


def run_example() -> None:
    _, counts, _ = run_workflow()
    plot_mappings(counts)


if __name__ == "__main__":
    run_example()
