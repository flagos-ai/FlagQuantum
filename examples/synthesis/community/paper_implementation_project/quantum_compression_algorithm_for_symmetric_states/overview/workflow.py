"""Execute all compression cases and perform the original finite sampling."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .quantum import Scenario, build_program

SHOTS = 2048
SCENARIOS: tuple[Scenario, ...] = ("uniform", "one_excitation", "nonsymmetric")


@dataclass(frozen=True)
class CompressionResult:
    """The actual native state, named-register distribution, and sampled counts."""

    scenario: Scenario
    program: synth.Program
    state: torch.Tensor
    probabilities: tuple[float, ...]
    counts: tuple[int, ...]


def run_case(scenario: Scenario, seed: int) -> CompressionResult:
    """Execute the native qfunc before drawing from its named-register readout."""
    program = build_program(scenario)
    result = fq.run(program)
    state = result.statevector().flatten()
    distribution = program.probabilities(result, registers=("x",))
    probabilities = tuple(
        float(distribution.get((word,), 0.0)) for word in range(state.numel())
    )
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = tuple(
        int(count) for count in torch.bincount(draws, minlength=len(probabilities))
    )
    return CompressionResult(scenario, program, state, probabilities, counts)
