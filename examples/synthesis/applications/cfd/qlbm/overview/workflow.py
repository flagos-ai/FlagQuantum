"""Run the native lattice workflow and draw the original number of samples."""

from dataclasses import dataclass
from typing import TypeAlias

import torch

import flagquantum as fq
from flagquantum import synth

from ..main import main
from ..physics import VELOCITY_AMPLITUDES, X_AMPLITUDES, Y_AMPLITUDES
from .model import GRID, MAGNITUDES, SHOTS, active_magnitudes

REGISTER_NAMES = ("x", "y", "vx", "vy", "ux", "uy")
Key: TypeAlias = tuple[int, ...]


@dataclass(frozen=True)
class Run:
    """Actual public output distribution and derived sampled marginals."""

    program: synth.Program
    probabilities: dict[Key, float]
    counts: dict[Key, int]
    position: torch.Tensor
    magnitude: torch.Tensor


def build_program() -> synth.Program:
    """Reuse the source-sized native quantum kernel with the fresh CFL schedule."""
    return synth.synthesize(
        main,
        classical_inputs={
            "x_amplitudes": X_AMPLITUDES,
            "y_amplitudes": Y_AMPLITUDES,
            "ux_amplitudes": VELOCITY_AMPLITUDES,
            "uy_amplitudes": VELOCITY_AMPLITUDES,
            "active": active_magnitudes(),
        },
        max_qubits=20,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )


def run_workflow(seed: int = 0) -> Run:
    """Sample joint output, then derive both source marginal distributions."""
    program = build_program()
    result = fq.run(program)
    distribution = program.probabilities(result, registers=REGISTER_NAMES)
    probabilities = {word: float(weight) for word, weight in distribution.items()}
    keys = tuple(sorted(probabilities))
    weights = torch.tensor([probabilities[key] for key in keys], dtype=torch.float64)
    draw = torch.multinomial(
        weights, SHOTS, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    observed = torch.bincount(draw, minlength=len(keys))
    counts = {
        key: int(count) for key, count in zip(keys, observed, strict=True) if count
    }
    position = torch.zeros((GRID, GRID), dtype=torch.float64)
    magnitude = torch.zeros((len(MAGNITUDES), len(MAGNITUDES)), dtype=torch.float64)
    for (x, y, _, _, ux, uy), count in counts.items():
        position[y, x] += count / SHOTS
        magnitude[uy, ux] += count / SHOTS
    return Run(program, probabilities, counts, position, magnitude)
