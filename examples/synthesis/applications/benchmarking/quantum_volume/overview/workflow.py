"""Original forty-circuit experiment, independent reference and observed-count score."""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .ansatz import Schedule, build_program, make_schedule


@dataclass(frozen=True)
class Trial:
    size: int
    index: int
    seed: int
    counts: tuple[tuple[int, int], ...]
    observed_heavy_count: int
    ideal_heavy_probability: float
    state_error: float
    named_probability_error: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class SizeResult:
    size: int
    observed_heavy_probability: float
    source_standard_deviation: float
    lower_bound: float
    passes_source_threshold: bool
    mean_ideal_heavy_probability: float


@dataclass(frozen=True)
class Experiment:
    trials: tuple[Trial, ...]
    sizes: tuple[SizeResult, ...]
    source_volume: int
    programs: tuple[synth.Program, ...] = ()


def circuit_reference(size: int, schedule: Schedule) -> torch.Tensor:
    """Apply the fresh matrices directly, with explicit least-significant list order."""
    circuit = fq.Circuit(size, dtype=torch.complex128)
    for first, second, matrix in schedule:
        circuit.any(
            size - 1 - second,
            size - 1 - first,
            matrix=torch.tensor(matrix, dtype=torch.complex128),
        )
    state = circuit.state()
    assert isinstance(state, torch.Tensor)
    return state.flatten()


def dense_reference(size: int, schedule: Schedule) -> NDArray[np.complex128]:
    """Independently embed each local matrix in integer basis order."""
    state = np.zeros(1 << size, dtype=np.complex128)
    state[0] = 1
    for first, second, matrix in schedule:
        updated = np.zeros_like(state)
        first_bit = 1 << first
        second_bit = 1 << second
        for column, amplitude in enumerate(state):
            local = int(bool(column & first_bit)) + 2 * int(bool(column & second_bit))
            untouched = column & ~(first_bit | second_bit)
            for row in range(4):
                output = (
                    untouched
                    | ((row & 1) * first_bit)
                    | (((row >> 1) & 1) * second_bit)
                )
                updated[output] += matrix[row][local] * amplitude
        state = updated
    return state


def observed_heavy_count(counts: tuple[tuple[int, int], ...]) -> int:
    """Retain the source median of positive observed counts, not the ideal median."""
    if not counts or any(
        type(word) is not int or word < 0 or type(count) is not int or count <= 0
        for word, count in counts
    ):
        raise ValueError("use nonempty positive observed integer counts")
    if len({word for word, _ in counts}) != len(counts):
        raise ValueError("observed words must be unique")
    median = float(np.median([count for _, count in counts]))
    return sum(count for _, count in counts if count >= median)


def source_round_significant(value: float) -> float:
    """Keep the source one-significant-digit display, including a defined zero case."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("standard deviation must be nonnegative and finite")
    return 0.0 if value == 0 else round(value, -math.floor(math.log10(value)))


def check_and_sample(
    program: synth.Program, size: int, schedule: Schedule, *, index: int, seed: int
) -> Trial:
    result = fq.run(program)
    state = result.statevector()
    assert isinstance(state, torch.Tensor)
    actual = state.flatten()
    expected = circuit_reference(size, schedule)
    classical = torch.from_numpy(dense_reference(size, schedule))
    error = float(
        max((actual - expected).abs().max(), (expected - classical).abs().max())
    )
    marginal = program.probabilities(result, registers=("target",))
    probabilities = torch.zeros(1 << size, dtype=torch.float64)
    for key, probability in marginal.items():
        probabilities[int(key[0])] = probability
    # Array element zero is the least significant bit of each named word.
    expected_words = expected.abs().square()
    named_error = float((probabilities - expected_words).abs().max())
    if error > 2e-11 or named_error > 2e-11 or actual.numel() != 1 << size:
        raise AssertionError(
            "full state, global phase, named bit order or workspace differs"
        )
    counts = torch.bincount(
        torch.multinomial(
            probabilities,
            100,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=1 << size,
    )
    observed = tuple(
        (word, int(count)) for word, count in enumerate(counts) if int(count)
    )
    ideal = expected.abs().square()
    median = (
        float(ideal.median()) if ideal.numel() % 2 else float(np.median(ideal.numpy()))
    )
    ideal_heavy = float(ideal[ideal > median].sum())
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Trial(
        size,
        index,
        seed,
        observed,
        observed_heavy_count(observed),
        ideal_heavy,
        error,
        named_error,
        resources,
    )


def summarize(trials: tuple[Trial, ...]) -> Experiment:
    if len(trials) != 40 or {(t.size, t.index) for t in trials} != {
        (size, index) for size in range(3, 7) for index in range(10)
    }:
        raise ValueError("retain all forty source trials")
    sizes = []
    volume_power = 0
    accepting = True
    for size in range(3, 7):
        group = tuple(t for t in trials if t.size == size)
        values = np.array([t.observed_heavy_count for t in group], dtype=np.float64)
        probability = float(values.sum() / 1000)
        standard_deviation = float(values.std() / 1000)
        lower = probability - 2 * standard_deviation
        passed = lower >= 2 / 3
        sizes.append(
            SizeResult(
                size,
                probability,
                standard_deviation,
                lower,
                passed,
                sum(t.ideal_heavy_probability for t in group) / 10,
            )
        )
        if accepting and passed:
            volume_power = size
        else:
            accepting = False
    return Experiment(trials, tuple(sizes), 2**volume_power)


def run_experiment() -> Experiment:
    """Run original sizes 3..6, ten fresh circuits per size and 100 actual draws."""
    rng = np.random.RandomState(0)
    permutations = random.Random(0)
    trials = []
    programs = []
    for size in range(3, 7):
        for index in range(10):
            schedule = make_schedule(size, rng, permutations)
            program = build_program(size, schedule)
            programs.append(program)
            trials.append(
                check_and_sample(
                    program,
                    size,
                    schedule,
                    index=index,
                    seed=202600 + 10 * size + index,
                )
            )
    summary = summarize(tuple(trials))
    return Experiment(
        summary.trials, summary.sizes, summary.source_volume, tuple(programs)
    )
