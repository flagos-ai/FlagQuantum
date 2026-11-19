"""Eight-signal BB84 teaching workflow with explicitly bound protocol inputs."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import (
    ALICE_BASES,
    BOB_BASES,
    MESSAGE,
    build_program,
)
from flagquantum import synth


@dataclass(frozen=True)
class Observation:
    """One actually sampled eight-signal result in logical signal order."""

    word: int
    bits: tuple[int, ...]
    count: int


@dataclass(frozen=True)
class KeyReport:
    """Source teaching postprocessing; parity pairs do not certify key security."""

    matching_positions: tuple[int, ...]
    alice_raw_key: tuple[int, ...]
    bob_raw_key: tuple[int, ...]
    error_rate: float
    corrected_key: tuple[int, ...]
    final_key: tuple[int, ...] | None


@dataclass(frozen=True)
class Workflow:
    """Actual native execution, observations and modal-signal teaching report."""

    program: synth.Program
    observations: tuple[Observation, ...]
    selected: Observation
    report: KeyReport


def _binary(values: tuple[int, ...], name: str) -> None:
    if len(values) != 8 or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError(f"{name} requires exactly eight binary integer values")


def postprocess(
    bob_measurements: tuple[int, ...],
    *,
    message: tuple[int, ...],
    alice_bases: tuple[int, ...],
    bob_bases: tuple[int, ...],
) -> KeyReport:
    """Retain source reconciliation, error threshold and pairwise parity teaching steps."""
    for name, values in (
        ("bob_measurements", bob_measurements),
        ("message", message),
        ("alice_bases", alice_bases),
        ("bob_bases", bob_bases),
    ):
        _binary(values, name)
    positions = tuple(i for i in range(8) if alice_bases[i] == bob_bases[i])
    alice_raw = tuple(message[i] for i in positions)
    bob_raw = tuple(bob_measurements[i] for i in positions)
    if not positions:
        return KeyReport(positions, alice_raw, bob_raw, 1.0, (), None)
    errors = sum(a != b for a, b in zip(alice_raw, bob_raw, strict=True))
    rate = errors / len(positions)
    if rate >= 0.15:
        return KeyReport(positions, alice_raw, bob_raw, rate, (), None)
    corrected = tuple(a for a, b in zip(alice_raw, bob_raw, strict=True) if a == b)
    parity = tuple(
        corrected[i] ^ corrected[i + 1] for i in range(0, len(corrected) - 1, 2)
    )
    return KeyReport(positions, alice_raw, bob_raw, rate, corrected, parity)


def observe(
    program: synth.Program, *, shots: int = 2048, seed: int = 3
) -> tuple[Observation, ...]:
    """Draw final computational-basis observations from actual named native readout."""
    if (
        isinstance(shots, bool)
        or not isinstance(shots, int)
        or not 1 <= shots <= 100000
    ):
        raise ValueError("shots must be an integer in [1, 100000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if tuple(program.registers) != ("qba",) or len(program.registers["qba"]) != 8:
        raise ValueError("BB84 requires one eight-bit qba register")
    distribution = dict(program.probabilities(fq.run(program), registers=("qba",)))
    if not set(distribution).issubset({(word,) for word in range(256)}):
        raise ValueError("BB84 readout is outside the eight-signal domain")
    probabilities = tuple(distribution.get((word,), 0.0) for word in range(256))
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-10
    ):
        raise ValueError("BB84 readout must be a normalized nonnegative distribution")
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=256)
    return tuple(
        sorted(
            (
                Observation(
                    word, tuple((word >> i) & 1 for i in range(8)), int(counts[word])
                )
                for word in range(256)
                if int(counts[word]) > 0
            ),
            key=lambda row: (-row.count, row.word),
        )
    )


def run_workflow(
    *,
    message: tuple[int, ...] = MESSAGE,
    alice_bases: tuple[int, ...] = ALICE_BASES,
    bob_bases: tuple[int, ...] = BOB_BASES,
    shots: int = 2048,
    seed: int = 3,
) -> Workflow:
    """Prepare the original eight signals, observe and use the same protocol inputs."""
    program = build_program(message, alice_bases, bob_bases)
    observations = observe(program, shots=shots, seed=seed)
    selected = observations[0]
    report = postprocess(
        selected.bits, message=message, alice_bases=alice_bases, bob_bases=bob_bases
    )
    return Workflow(program, observations, selected, report)
