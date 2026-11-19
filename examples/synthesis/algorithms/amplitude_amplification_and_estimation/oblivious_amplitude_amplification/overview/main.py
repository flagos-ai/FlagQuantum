"""Complete local stages of the source oblivious-amplification Notebook."""

from dataclasses import dataclass

import numpy as np

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.main import (
    build_program,
)
from flagquantum import synth


@dataclass(frozen=True)
class Stage:
    """Named joint output and its block-zero marginal from actual execution."""

    program: synth.Program
    probabilities: tuple[tuple[tuple[int, int], float], ...]
    good_probability: float

    def good_rows(self) -> tuple[tuple[int, float], ...]:
        """Return source-style data/probability rows with block equal to zero."""
        return tuple(
            (data, probability)
            for (data, block), probability in self.probabilities
            if block == 0
        )


def source_amplitudes() -> tuple[float, ...]:
    """Retain the source's independent seed-one four-amplitude preparation."""
    values = np.random.RandomState(1).rand(4)
    return tuple(float(value) for value in values / np.linalg.norm(values))


def run_stage(repetitions: int, amplitudes: tuple[float, ...]) -> Stage:
    """Execute the existing native composition and sum the named good block."""
    program = build_program(repetitions, amplitudes)
    probabilities = program.probabilities(fq.run(program), registers=("data", "block"))
    rows = tuple(
        sorted(((data, block), p) for (data, block), p in probabilities.items())
    )
    return Stage(program, rows, sum(value for (_, block), value in rows if block == 0))


def run_workflow() -> tuple[Stage, Stage]:
    """Run both source experiments; certify good-block weight, not just norm."""
    amplitudes = source_amplitudes()
    before = run_stage(0, amplitudes)
    after = run_stage(1, amplitudes)
    assert np.isclose(before.good_probability, 0.25, rtol=0, atol=1e-12)
    assert np.isclose(after.good_probability, 1.0, rtol=0, atol=1e-12)
    for name, stage in (("Block encoding", before), ("One OAA iteration", after)):
        print(name)
        print("Good data/probability rows:", stage.good_rows())
        print("Fraction of good states:", stage.good_probability)
        print("Total probability:", sum(p for _, p in stage.probabilities))
        print("Resources:", stage.program.report["resources"])
    return before, after


def run_example() -> None:
    run_workflow()


if __name__ == "__main__":
    run_example()
