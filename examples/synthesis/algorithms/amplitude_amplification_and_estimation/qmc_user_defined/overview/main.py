"""Complete native quantum Monte Carlo integration workflow."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import torch

import flagquantum as fq
from flagquantum import synth

from .quantum import estimation_program, grover_program, preparation_program

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PROBABILITIES = tuple(i / 28 for i in range(8))


def build_program(
    stage: Literal["preparation", "grover", "estimation"],
    probabilities: tuple[float, ...] = PROBABILITIES,
    slope: float = 0.5,
    offset: float = 0.4,
) -> synth.Program:
    """Bind actual source-precision probabilities rather than frozen rounded rows."""
    if (
        len(probabilities) != 8
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0
            for p in probabilities
        )
        or abs(sum(probabilities) - 1) > 1e-12
        or type(slope) not in (int, float)
        or not math.isfinite(slope)
        or type(offset) not in (int, float)
        or not math.isfinite(offset)
    ):
        raise ValueError(
            "Use eight normalized finite nonnegative probabilities and finite real angles."
        )
    functions = {
        "preparation": preparation_program,
        "grover": grover_program,
        "estimation": estimation_program,
    }
    if stage not in functions:
        raise ValueError("Choose preparation, grover or estimation.")
    return synth.synthesize(
        functions[stage],
        classical_inputs={
            "probabilities": list(probabilities),
            "angles": [slope * i + offset for i in range(8)],
        },
        max_qubits=7,
        max_workspace=0,
    )


@dataclass(frozen=True)
class Workflow:
    """Three actual native programs and their finite source-profile readout."""

    preparation: synth.Program
    grover: synth.Program
    estimation: synth.Program
    counts: tuple[tuple[Fraction, int], ...]
    selected_phase: Fraction
    measured_amplitude: float
    exact_amplitude: float


def read_estimation(
    preparation: synth.Program, grover: synth.Program, estimation: synth.Program
) -> Workflow:
    """Execute all three stages and sample the actual signed phase distribution."""
    fq.run(preparation)
    fq.run(grover)
    probabilities = estimation.numeric_probabilities(
        fq.run(estimation), registers=("phase_result",), threshold=0
    )
    keys = tuple(key[0] for key in probabilities)
    samples = torch.multinomial(
        torch.tensor(list(probabilities.values()), dtype=torch.float64),
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(123),
    )
    frequencies = torch.bincount(samples, minlength=8)
    counts = tuple((key, int(n)) for key, n in zip(keys, frequencies, strict=True))
    selected = keys[int(frequencies.argmax())]
    measured = math.sin(math.pi * float(selected)) ** 2
    exact = sum(math.sin(0.25 * i + 0.2) ** 2 * p for i, p in enumerate(PROBABILITIES))
    print("Most frequent signed phase:", selected)
    print("Measured amplitude:", measured, "exact amplitude:", exact)
    print("Local phase samples:", counts)
    for name, program in (
        ("preparation", preparation),
        ("Grover", grover),
        ("QPE", estimation),
    ):
        print(name, "resources:", program.report["resources"])
    if abs(measured - exact) >= 0.1:
        raise ValueError("Three-bit sampled amplitude misses the source tolerance.")
    return Workflow(preparation, grover, estimation, counts, selected, measured, exact)


def run_workflow() -> Workflow:
    return read_estimation(
        build_program("preparation"),
        build_program("grover"),
        build_program("estimation"),
    )


def plot_results(workflow: Workflow) -> "Figure":
    import matplotlib.pyplot as plt

    rows = sorted(workflow.counts)
    figure, axes = plt.subplots(figsize=(8, 4), layout="constrained")
    axes.bar([float(x) for x, _ in rows], [count for _, count in rows], width=0.1)
    axes.set_xlabel("Signed fractional phase")
    axes.set_ylabel("Count (2048 local samples)")
    axes.tick_params(axis="x", labelrotation=90)
    plt.show()
    return figure


def run_example() -> None:
    plot_results(run_workflow())


if __name__ == "__main__":
    run_example()
