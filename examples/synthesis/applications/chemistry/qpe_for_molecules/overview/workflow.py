"""Run signed seven-bit LiH phase estimation with fresh coefficients."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.qpe_for_molecules.physics import (
    ZERO_STATE,
    commuting_runs,
    observable,
)
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    inplace_prepare_amplitudes,
    qft,
    qfunc,
    suzuki_trotter,
)

from .model import WIDTH, Model, build_model, gamma_two

PHASE_BITS = 7
PHASE_COUNT = 1 << PHASE_BITS
POWERS = tuple(2**bit for bit in range(PHASE_BITS))
EPS = 0.1
EXPERIMENTAL_R0 = 0.05
STEPS = tuple(math.ceil(EXPERIMENTAL_R0 * p**1.5) for p in POWERS)
SHOTS = 1000
SEED = 246
ASSUMED_OVERLAP = 0.05


@dataclass(frozen=True)
class Report:
    model: Model
    program: synth.Program
    gamma: float
    adaptive_bound: tuple[int, ...]
    naive_bound: tuple[int, ...]
    phases: tuple[float, ...]
    energies: tuple[float, ...]
    counts: tuple[int, ...]
    exact: tuple[float, ...]
    peaks: tuple[float, ...]
    mode: float
    lowest_peak: float


def build_program(
    model: Model, scale: float = 1.0, amplitudes: tuple[float, ...] = ZERO_STATE
) -> synth.Program:
    """Compose separately specialized Strang evolutions under phase controls."""
    if not math.isfinite(scale) or not 0 < scale <= 2:
        raise ValueError("The positive time scale must not exceed two")
    if len(amplitudes) != 1 << WIDTH or not math.isclose(
        math.fsum(value * value for value in amplitudes), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("The six-bit state amplitudes must be normalized")
    g0, g1, g2, g3, g4, g5, g6, g7 = map(observable, commuting_runs(model.terms))

    @qfunc
    def power(time: float, n: int, state: QArray[QBit]) -> None:
        suzuki_trotter(g0, time / (2 * n), 1, 1, state)
        for i in range(n):
            suzuki_trotter(g1, time / (2 * n), 1, 1, state)
            suzuki_trotter(g2, time / (2 * n), 1, 1, state)
            suzuki_trotter(g3, time / (2 * n), 1, 1, state)
            suzuki_trotter(g4, time / (2 * n), 1, 1, state)
            suzuki_trotter(g5, time / (2 * n), 1, 1, state)
            suzuki_trotter(g6, time / (2 * n), 1, 1, state)
            suzuki_trotter(g7, time / n, 1, 1, state)
            suzuki_trotter(g6, time / (2 * n), 1, 1, state)
            suzuki_trotter(g5, time / (2 * n), 1, 1, state)
            suzuki_trotter(g4, time / (2 * n), 1, 1, state)
            suzuki_trotter(g3, time / (2 * n), 1, 1, state)
            suzuki_trotter(g2, time / (2 * n), 1, 1, state)
            suzuki_trotter(g1, time / (2 * n), 1, 1, state)
            if i < n - 1:
                suzuki_trotter(g0, time / n, 1, 1, state)
            else:
                suzuki_trotter(g0, time / (2 * n), 1, 1, state)

    @qfunc
    def selected_power(time: float, n: int, bit: QBit, state: QArray[QBit]) -> None:
        control(bit, lambda: power(time, n, state))

    @qfunc
    def main(
        times: list[float],
        steps: list[int],
        amplitudes: list[float],
        state: Output[QArray[QBit]],
        phase: Output[QArray[QBit]],
    ) -> None:
        allocate(WIDTH, state)
        allocate(PHASE_BITS, phase)
        inplace_prepare_amplitudes(amplitudes, 0, state)
        hadamard_transform(phase)
        for i in range(PHASE_BITS):
            selected_power(times[i], steps[i], phase[i], state)
        with adjoint():
            qft(phase)

    return synth.synthesize(
        main,
        classical_inputs={
            "times": [-2 * math.pi * scale * p for p in POWERS],
            "steps": list(STEPS),
            "amplitudes": list(amplitudes),
        },
        max_qubits=13,
        max_workspace=0,
        max_operations=100_000,
        memory_limit_bytes=64 * 1024**2,
    )


def peak_indices(probabilities: tuple[float, ...], height: float) -> tuple[int, ...]:
    """Find interior local maxima, using the lower midpoint of tied plateaus."""
    peaks = []
    left = 1
    while left < len(probabilities) - 1:
        right = left
        while (
            right < len(probabilities) - 1
            and probabilities[right + 1] == probabilities[left]
        ):
            right += 1
        if (
            right < len(probabilities) - 1
            and probabilities[left] >= height
            and probabilities[left] > probabilities[left - 1]
            and probabilities[right] > probabilities[right + 1]
        ):
            peaks.append((left + right) // 2)
        left = right + 1
    return tuple(peaks)


def read_run(model: Model, program: synth.Program) -> Report:
    gamma = gamma_two(model.terms)
    r0 = math.sqrt(2**5 * gamma / (3 * EPS)) * (2 * math.pi) ** 1.5
    execution = fq.run(program)
    probabilities = program.probabilities(execution, registers=("phase",), threshold=0)
    words = tuple(range(PHASE_COUNT // 2, PHASE_COUNT)) + tuple(range(PHASE_COUNT // 2))
    phases = tuple(
        (word if word < PHASE_COUNT // 2 else word - PHASE_COUNT) / PHASE_COUNT
        for word in words
    )
    energies = tuple(value * model.normalization + model.offset for value in phases)
    exact = tuple(probabilities.get((word,), 0.0) for word in words)
    draws = torch.multinomial(
        torch.tensor(exact, dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    counts = tuple(map(int, torch.bincount(draws, minlength=PHASE_COUNT)))
    mode = max(range(PHASE_COUNT), key=counts.__getitem__)
    # Reproduce the sampled table: phase bins without observations are absent.
    observed = tuple(i for i, count in enumerate(counts) if count)
    peaks = tuple(
        observed[i]
        for i in peak_indices(
            tuple(counts[j] / SHOTS for j in observed), ASSUMED_OVERLAP * 0.4
        )
    )
    return Report(
        model,
        program,
        gamma,
        tuple(math.ceil(r0 * p**1.5) for p in POWERS),
        tuple(math.ceil(r0 * p) for p in POWERS),
        phases,
        energies,
        counts,
        exact,
        tuple(energies[i] for i in peaks),
        energies[mode],
        energies[peaks[0] if peaks else mode],
    )


def run_workflow() -> Report:
    model = build_model()
    return read_run(model, build_program(model))
