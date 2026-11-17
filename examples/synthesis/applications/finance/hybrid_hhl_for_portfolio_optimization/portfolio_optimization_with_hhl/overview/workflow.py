"""Use measured QPE bins to choose the reduced inverse-rotation table."""

import math
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from .model import PHASE_BITS, SHOTS, Model, build_model, reduced_table
from .quantum import build_hhl, spectroscopy, state


@dataclass(frozen=True)
class Spectrum:
    words: tuple[int, ...]
    counts: tuple[int, ...]
    eigenvalues: tuple[float, ...]
    projections: tuple[float, ...]
    selected_words: tuple[int, ...]


@dataclass(frozen=True)
class Profile:
    name: str
    counts: tuple[int, ...]
    overlap: float | None
    resources: ResourceEstimate
    target_identity: str
    projective_error: float

    @property
    def postselection_rate(self) -> float:
        return sum(self.counts[2:]) / SHOTS


def sample(
    program: synth.Program, registers: tuple[str, ...], seed: int
) -> tuple[int, ...]:
    """Draw the original finite shot count from the native execution result."""
    distribution = program.probabilities(fq.run(program), registers=registers)
    size = 1 << sum(len(program.registers[name]) for name in registers)
    weights = torch.zeros(size, dtype=torch.float64)
    for words, probability in distribution.items():
        index = 0
        for name, word in zip(registers, words, strict=True):
            index = (index << len(program.registers[name])) | word
        weights[index] = probability
    draws = torch.multinomial(
        weights, SHOTS, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    return tuple(map(int, torch.bincount(draws, minlength=size)))


def infer_spectrum(model: Model, counts: tuple[int, ...]) -> Spectrum:
    if len(counts) != 1 << PHASE_BITS or sum(counts) != SHOTS or min(counts) < 0:
        raise ValueError("Spectrum requires the original four-bit shot histogram")
    words = np.array(
        sorted(
            (i for i, count in enumerate(counts) if count),
            key=lambda i: (-counts[i], i),
        )
    )
    if len(words) < 4:
        raise ValueError("Spectrum requires four observed phase rows")
    probabilities = np.asarray(counts)[words] / SHOTS
    eigenvalues = words / ((1 << PHASE_BITS) * model.scale) - model.eigenvalue_bound
    projections = np.sqrt(probabilities) / eigenvalues
    maxima = np.argpartition(probabilities, -4)[-4:][::-1]
    # The source uses the offsets returned here directly against the sampled rows.
    selected = np.argpartition(abs(projections[maxima]), -2)[-2:][::-1]
    return Spectrum(
        tuple(map(int, words)),
        counts,
        tuple(map(float, eigenvalues)),
        tuple(map(float, projections)),
        tuple(int(words[index]) for index in selected),
    )


def conditional_overlap(counts: tuple[int, ...]) -> float | None:
    """Read the conditional SWAP overlap from indicator-one finite counts."""
    accepted = counts[2] + counts[3]
    radicand = 2 * counts[2] / accepted - 1 if accepted else -1.0
    return math.sqrt(radicand) if radicand >= 0 else None


def profile(name: str, program: synth.Program, counts: tuple[int, ...]) -> Profile:
    """Count a public CX/U decomposition after checking its global-phase orbit."""
    conversion = convert_basis(
        program.circuit, gates=("cx", "u"), max_added_operations=8192
    )
    converted = fq.run(conversion.program).statevector().flatten()
    native = state(program)
    aligned = converted * torch.vdot(converted, native)
    torch.testing.assert_close(aligned, native, atol=1e-12, rtol=0)
    return Profile(
        name,
        counts,
        conditional_overlap(counts),
        estimate_resources(conversion.program),
        conversion.conversion_identity,
        float((aligned - native).abs().max()),
    )


def run_experiment() -> tuple[Model, Spectrum, tuple[Profile, ...]]:
    model = build_model()
    spectrum = infer_spectrum(model, sample(spectroscopy(model), ("phase",), 9121))
    reduced = reduced_table(model, spectrum.selected_words)
    depth = build_hhl(model, reduced, "depth")
    width = build_hhl(model, reduced, "width")
    basic = build_hhl(model, model.basic_angles, "width")
    depth_counts = sample(depth, ("indicator", "test"), 9122)
    basic_counts = sample(basic, ("indicator", "test"), 9123)
    return (
        model,
        spectrum,
        (
            profile("Hybrid HHL (depth)", depth, depth_counts),
            profile("Hybrid HHL (width)", width, depth_counts),
            profile("Basic HHL", basic, basic_counts),
        ),
    )
