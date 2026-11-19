"""Full physical checks retain every dropped register as a private environment."""

from dataclasses import dataclass
from fractions import Fraction

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .ansatz import source_eigenpair
from .reference import reference_circuit


@dataclass(frozen=True)
class Outcome:
    word: int
    count: int
    probability: float


@dataclass(frozen=True)
class Check:
    comparison: bool
    state_error: float
    named_error: float
    resources: ResourceEstimate
    outcomes: tuple[Outcome, ...]
    exact_eigenvalue: float
    most_likely_phase: float | None
    observed_overlap: float | None
    exact_overlap: float | None
    seed: int
    shots: int


def verify_and_sample(
    program: synth.Program,
    *,
    comparison: bool = False,
    seed: int = 4100,
    shots: int = 1024,
) -> Check:
    if (
        type(seed) is not int
        or not 0 <= seed < 2**63
        or type(shots) is not int
        or not 1 <= shots <= 100000
    ):
        raise ValueError("use a nonnegative 63-bit seed and 1..100000 shots")
    result = fq.run(program)
    actual = result.statevector()
    expected = reference_circuit(comparison).state()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    error = float((actual.flatten() - expected.flatten()).abs().max())
    marginal = (
        expected.reshape(2, -1).abs().square().sum(dim=1)
        if comparison
        else expected.reshape(32, 4).abs().square().sum(dim=1)
    )
    if comparison:
        named = program.probabilities(result, registers=("test",))
        observed = torch.tensor(
            [named.get((word,), 0.0) for word in range(2)], dtype=torch.float64
        )
    else:
        decoded = program.numeric_probabilities(result, registers=("phase_approx",))
        observed = torch.tensor(
            [decoded.get((Fraction(word, 32),), 0.0) for word in range(32)],
            dtype=torch.float64,
        )
    named_error = float((observed - marginal).abs().max())
    if (
        error > 2e-11
        or named_error > 2e-11
        or actual.numel() != (8192 if comparison else 128)
    ):
        raise AssertionError("absolute physical state, phase or named values differ")
    counts = torch.bincount(
        torch.multinomial(
            observed,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(observed),
    )
    outcomes = tuple(
        Outcome(word, int(count), float(observed[word]))
        for word, count in enumerate(counts)
        if int(count)
    )
    exact_eigenvalue, _ = source_eigenpair()
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Check(
        comparison,
        error,
        named_error,
        resources,
        outcomes,
        exact_eigenvalue,
        None if comparison else int(counts.argmax()) / 32,
        2 * int(counts[0]) / shots - 1 if comparison else None,
        2 * float(marginal[0]) - 1 if comparison else None,
        seed,
        shots,
    )
