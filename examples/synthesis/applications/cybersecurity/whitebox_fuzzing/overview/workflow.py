"""Complete classical and quantum whitebox sampling with independent full states."""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from ..comparisons.circuit import reference_circuit
from .operations import build_program

if TYPE_CHECKING:
    from matplotlib.figure import Figure

Label = Literal["a", "b", "c", "d", "e"]
LABELS: tuple[Label, ...] = ("a", "b", "c", "d", "e")


def foo(x: int, y: int) -> Label:
    """Retain the original classical program and its branch priority."""
    if x == 12:
        return "a" if y > 3 else "b"
    if y + x < 9:
        return "c" if (x % 4) * (y % 2) == 1 else "d"
    return "e"


@dataclass(frozen=True)
class Observation:
    x: int
    y: int
    count: int
    probability: float
    output: Label


@dataclass(frozen=True)
class StateCheck:
    iterations: int | None
    maximum_state_error: float
    dirty_workspace_probability: float
    reference_dirty_probability: float
    resources: ResourceEstimate


def classical_samples() -> tuple[Observation, ...]:
    """Preserve source RandomState(3) and two 500-element draws in 0..62."""
    rng = np.random.RandomState(3)
    xs, ys = rng.randint(0, 63, 500), rng.randint(0, 63, 500)
    counts = Counter((int(x), int(y)) for x, y in zip(xs, ys, strict=True))
    return tuple(
        Observation(x, y, count, count / 500, foo(x, y))
        for (x, y), count in sorted(counts.items())
    )


def source_recommendation() -> int:
    """Report the source heuristic using two marked inputs, independently of the actual ten rounds."""
    return int(np.round(np.pi / 4 * np.sqrt(2**12 / 2)))


def marked_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (x, y)
        for x in range(64)
        for y in range(64)
        if x + y < 9 and (x % 4) * (y % 2) == 1
    )


def independent_state(iterations: int | None) -> tuple[torch.Tensor, float]:
    """Use the retained independent Circuit and a separate sign truth table."""
    if iterations is None:
        circuit = fq.Circuit(12, dtype=torch.complex128)
        for bit in range(12):
            circuit.h(bit)
        raw = fq.run(circuit).statevector()
        assert isinstance(raw, torch.Tensor)
        state = raw.flatten().clone()
        for x, y in marked_pairs():
            state[(x << 6) | y] *= -1
        return state, 0.0
    raw = fq.run(reference_circuit(iterations)).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(4096, -1)
    dirty = float(state[:, 1:].abs().square().sum())
    assert dirty < 1e-22
    return state[:, 0], dirty


def verify(program: synth.Program, iterations: int | None) -> StateCheck:
    expected, reference_dirty = independent_state(iterations)
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.flatten()
    width = state.numel().bit_length() - 1
    logical = (*program.registers["x"], *program.registers["y"])
    hidden = tuple(q for q in range(width) if q not in logical)
    ordered = state.reshape((2,) * width).permute((*logical, *hidden)).reshape(4096, -1)
    error = float((ordered[:, 0] - expected).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    torch.testing.assert_close(ordered[:, 0], expected, atol=3e-11, rtol=0)
    assert dirty < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.n_qubits <= 25
    return StateCheck(iterations, error, dirty, reference_dirty, resources)


def quantum_samples(
    program: synth.Program, seed: int = 1913
) -> tuple[Observation, ...]:
    """Take 500 actual local draws from named x/y probabilities, with no bit decoding."""
    probabilities = dict(program.probabilities(fq.run(program), registers=("x", "y")))
    labels = tuple(probabilities)
    weights = np.array([probabilities[key] for key in labels], dtype=float)
    weights /= weights.sum()
    counts = np.random.default_rng(seed).multinomial(500, weights)
    return tuple(
        Observation(
            int(key[0]),
            int(key[1]),
            int(count),
            float(weight),
            foo(int(key[0]), int(key[1])),
        )
        for key, count, weight in zip(labels, counts, weights, strict=True)
        if count
    )


def output_counts(observations: tuple[Observation, ...]) -> dict[Label, int]:
    result: dict[Label, int] = dict.fromkeys(LABELS, 0)
    for row in observations:
        result[row.output] += row.count
    return result


def plot_outputs(observations: tuple[Observation, ...], title: str) -> "Figure":
    """Show all five branch labels, including zero-count outcomes."""
    from matplotlib import pyplot as plt

    counts = output_counts(observations)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(tuple(counts), tuple(counts.values()))
    axis.set(xlabel="Output", ylabel="Occurrences", title=title)
    return figure


def run_example() -> tuple[StateCheck, ...]:
    classical = classical_samples()
    print("Classical500 samples:", output_counts(classical))
    print("Source heuristic:", source_recommendation(), "Actual search rounds:", 10)
    print("Actual marked pairs:", marked_pairs())
    checks: list[StateCheck] = []
    for iterations in (None, 0, 1, 2, 10):
        program = build_program(iterations)
        check = verify(program, iterations)
        checks.append(check)
        print(check)
        if iterations == 10:
            observations = quantum_samples(program)
            print("Local500 Grover samples:", output_counts(observations))
            print("Observed rows:", observations)
    return tuple(checks)
