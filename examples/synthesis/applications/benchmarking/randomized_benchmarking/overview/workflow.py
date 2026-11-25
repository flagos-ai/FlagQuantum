"""Bounded concurrent synthesis, explicit local simulator routes and tagged analysis."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import Output, QBit, qfunc

from .ansatz import LENGTHS, Sequence, fresh_sequences, make_model
from .reference import circuit_reference, dense_reference

Mode = Literal["statevector", "density_matrix"]
MODES: tuple[Mode, ...] = ("statevector", "density_matrix")


@dataclass(frozen=True)
class Trial:
    length: int
    mode: Mode
    seed: int
    shots: int
    counts: tuple[int, int]
    survival: float
    state_error: float
    representation_error: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class Experiment:
    sequences: tuple[Sequence, ...]
    programs: tuple[synth.Program, ...]
    trials: tuple[Trial, ...]
    batch_sizes: tuple[int, ...]
    source_label_mismatches: tuple[int, ...]


async def synthesize_all(
    models: tuple[Callable[[Output[QBit]], None], ...],
) -> tuple[synth.Program, ...]:
    """Preserve source asynchronous synthesis of its five independent models."""
    if len(models) != 5:
        raise ValueError("retain the five original native models")
    return tuple(
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    synth.synthesize,
                    qfunc(model),
                    max_qubits=1,
                    max_workspace=0,
                    memory_limit_bytes=1024**2,
                )
                for model in models
            )
        )
    )


def verify_program(program: synth.Program, gates: Sequence) -> float:
    actual = fq.run(program).statevector()
    expected = circuit_reference(gates)
    dense = torch.from_numpy(dense_reference(gates))
    error = float(
        max((actual.flatten() - expected).abs().max(), (expected - dense).abs().max())
    )
    named = program.probabilities(fq.run(program), registers=("target",))
    if error > 2e-12 or abs(named.get((0,), 0) - 1) > 2e-12 or actual.numel() != 2:
        raise AssertionError(
            "absolute inverse state, named target or workspace differs"
        )
    return error


def execute_trial(
    program: synth.Program, gates: Sequence, mode: Mode, *, shots: int, seed: int
) -> Trial:
    """Run the public synthesized IR in an explicitly selected numerical representation."""
    if (
        mode not in MODES
        or type(shots) is not int
        or not 1 <= shots <= 100000
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError(
            "use an admitted local mode, 1..100000 shots and nonnegative 63-bit seed"
        )
    state_error = verify_program(program, gates)
    options = fq.ExecutionOptions(
        mode=mode,
        backend="torch",
        device="cpu",
        precision="complex128",
        allow_approximate=False,
        allow_backend_fallback=False,
        memory_limit_bytes=1024**2,
    )
    exact = fq.run(
        program.circuit,
        options=options,
        outputs=fq.probabilities() if mode == "statevector" else fq.density_matrix(),
    )
    expected_state = circuit_reference(gates)
    if mode == "statevector":
        representation_error = float(
            (exact.statevector().flatten() - expected_state).abs().max()
        )
    else:
        expected_density = torch.outer(expected_state, expected_state.conj())
        representation_error = float(
            (exact.density_matrix.reshape(2, 2) - expected_density).abs().max()
        )
    if representation_error > 2e-12:
        raise AssertionError("the explicitly selected simulator representation differs")
    result = fq.run(
        program.circuit,
        options=fq.ExecutionOptions(
            mode=mode,
            backend="torch",
            device="cpu",
            precision="complex128",
            target="samples",
            shots=shots,
            seed=seed,
            allow_approximate=False,
            allow_backend_fallback=False,
            memory_limit_bytes=1024**2,
        ),
    )
    samples = result.require_samples().reshape(-1)
    if samples.numel() != shots or not bool(((samples == 0) | (samples == 1)).all()):
        raise AssertionError("actual readout shape or bit order differs")
    counts = (int((samples == 0).sum()), int((samples == 1).sum()))
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Trial(
        len(gates),
        mode,
        seed,
        shots,
        counts,
        counts[0] / shots,
        state_error,
        representation_error,
        resources,
    )


async def execute_batches(
    programs: tuple[synth.Program, ...],
    sequences: tuple[Sequence, ...],
    *,
    shots: int = 1024
) -> tuple[tuple[Trial, ...], tuple[int, ...]]:
    if len(programs) != len(sequences) or tuple(map(len, sequences)) != LENGTHS:
        raise ValueError("retain one program for each original sequence")
    jobs = [
        (program, gates, mode)
        for program, gates in zip(programs, sequences, strict=True)
        for mode in MODES
    ]
    results: list[Trial] = []
    batch_sizes = []
    for start in range(0, len(jobs), 3):
        batch = jobs[start : start + 3]
        batch_sizes.append(len(batch))
        results.extend(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        execute_trial,
                        program,
                        gates,
                        mode,
                        shots=shots,
                        seed=5100 + start + offset,
                    )
                    for offset, (program, gates, mode) in enumerate(batch)
                )
            )
        )
    return tuple(results), tuple(batch_sizes)


def source_label_mismatches(trials: tuple[Trial, ...]) -> tuple[int, ...]:
    """Expose the source's repeated-length labels versus its actual product order."""
    repeated = LENGTHS * len(MODES)
    if len(trials) != len(repeated):
        raise ValueError("retain all ten source execution jobs")
    return tuple(
        index
        for index, (trial, length) in enumerate(zip(trials, repeated, strict=True))
        if trial.length != length
    )


async def run_experiment_async() -> Experiment:
    sequences = fresh_sequences()
    programs = await synthesize_all(tuple(make_model(gates) for gates in sequences))
    trials, batch_sizes = await execute_batches(programs, sequences)
    return Experiment(
        sequences, programs, trials, batch_sizes, source_label_mismatches(trials)
    )


def run_experiment() -> Experiment:
    return asyncio.run(run_experiment_async())
