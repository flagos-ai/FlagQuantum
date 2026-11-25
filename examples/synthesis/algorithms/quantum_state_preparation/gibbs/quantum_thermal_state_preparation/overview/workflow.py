"""Sample both Fourier windows and the full thermal channel."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .model import FOURIER_SHOTS, SPECTRUM_BITS, THERMAL_SHOTS, gibbs_probabilities
from .quantum import build_probe, build_thermal


@dataclass(frozen=True)
class Shots:
    """Exact output weights and actual observed integer counts."""

    probabilities: tuple[float, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class Workflow:
    """Both source Fourier probes, the measured system, and the Gibbs reference."""

    gaussian: Shots
    uniform: Shots
    thermal: Shots
    exact_gibbs: tuple[float, ...]
    thermal_program: synth.Program


def _sample(
    program: synth.Program, register: str, n_words: int, shots: int, seed: int
) -> Shots:
    distribution = program.probabilities(fq.run(program), registers=(register,))
    probabilities = tuple(
        float(distribution.get((word,), 0.0)) for word in range(n_words)
    )
    sampled = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = tuple(int(value) for value in torch.bincount(sampled, minlength=n_words))
    return Shots(probabilities, counts)


def run_workflow() -> Workflow:
    """Execute and sample all three source circuits at their recorded shot counts."""
    words = 1 << SPECTRUM_BITS
    gaussian = _sample(
        build_probe(gaussian_window=True), "bohr", words, FOURIER_SHOTS, 7
    )
    uniform = _sample(
        build_probe(gaussian_window=False), "bohr", words, FOURIER_SHOTS, 8
    )
    program = build_thermal()
    thermal = _sample(program, "state", 4, THERMAL_SHOTS, 9)
    theoretical = gibbs_probabilities()
    for count, expected in zip(thermal.counts, theoretical, strict=True):
        if abs(count / THERMAL_SHOTS - expected) >= 0.1:
            raise ValueError("Thermal sample exceeds the original Gibbs tolerance")
    return Workflow(gaussian, uniform, thermal, theoretical, program)
