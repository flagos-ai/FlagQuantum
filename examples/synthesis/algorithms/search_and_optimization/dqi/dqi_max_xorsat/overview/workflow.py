"""Run the actual finite-shot unary, Dicke, and Max-XORSAT examples."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .model import DQI_SHOTS, UNARY_SHOTS, DQIModel, make_model
from .quantum import build_dicke, build_dqi, build_unary


@dataclass(frozen=True)
class Workflow:
    """Actual histograms, oracle reference scores, and the compiled main program."""

    model: DQIModel
    unary: tuple[tuple[int, int], ...]
    dicke_four: tuple[tuple[int, int], ...]
    dicke_two: tuple[tuple[int, int], ...]
    dqi: tuple[tuple[tuple[int, ...], int], ...]
    program: synth.Program
    mean_score: float


def _draw(
    program: synth.Program, registers: tuple[str, ...], shots: int, seed: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    probabilities = program.probabilities(fq.run(program), registers=registers)
    words = tuple(sorted(probabilities))
    weights = torch.tensor(
        [float(probabilities[word]) for word in words], dtype=torch.float64
    )
    draws = torch.multinomial(
        weights, shots, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    counts = torch.bincount(draws, minlength=len(words))
    return tuple(
        (word, int(count)) for word, count in zip(words, counts, strict=True) if count
    )


def _single(program: synth.Program, seed: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (word[0], count)
        for word, count in _draw(program, ("result",), UNARY_SHOTS, seed)
    )


def run_workflow() -> Workflow:
    """Regenerate the source tables, run all four circuits and check the sampled decoder."""
    model = make_model()
    unary = _single(build_unary(), 7)
    dicke_four = _single(build_dicke(4, use_unary=False), 8)
    dicke_two = _single(build_dicke(2, use_unary=True), 9)
    program = build_dqi(model)
    result = _draw(program, ("y", "solution"), DQI_SHOTS, 10)
    if any(word[0] != 0 for word, _ in result):
        raise ValueError("The decoder left a nonzero sampled error word")
    mean = sum(model.objective[word[1]] * count for word, count in result) / DQI_SHOTS
    return Workflow(model, unary, dicke_four, dicke_two, result, program, mean)
