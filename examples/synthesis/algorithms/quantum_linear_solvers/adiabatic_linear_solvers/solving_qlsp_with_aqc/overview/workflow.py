"""Run and compare each original adiabatic evolution scenario."""

from dataclasses import dataclass

import numpy as np

import flagquantum as fq
from flagquantum import synth

from .model import (
    PERTURBED_MATRIX,
    SOURCE_MATRIX,
    ComplexArray,
    LinearSystem,
    linear_system,
)
from .quantum import build_program

SCENARIOS = (("original", 7, 50), ("perturbed", 7, 50), ("refined", 10, 100))


@dataclass(frozen=True)
class EvolutionResult:
    """Actual synthesized resources, full state, and overlap with the direct solve."""

    program: synth.Program
    state: ComplexArray
    fidelity: float


def source_models() -> tuple[LinearSystem, LinearSystem]:
    """Construct both systems afresh from the source inputs."""
    return linear_system(SOURCE_MATRIX), linear_system(PERTURBED_MATRIX)


def evolve(model: LinearSystem, duration: int, steps: int) -> EvolutionResult:
    """Read all eight amplitudes in ascending basis-word order."""
    program = build_program(model, duration, steps)
    state = np.asarray(
        fq.run(program).statevector().detach().cpu().numpy(),
        dtype=np.complex128,
    ).reshape(8)
    return EvolutionResult(program, state, float(abs(np.vdot(state, model.exact)) ** 2))
