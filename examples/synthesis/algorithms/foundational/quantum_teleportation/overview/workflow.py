"""Native private-feedback teleportation with explicitly deferred measurement."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.quantum_teleportation.main import (
    build_program,
)
from flagquantum import synth


@dataclass(frozen=True)
class Workflow:
    """Bob's actual marginal state and final computational-basis observations."""

    program: synth.Program
    density: torch.Tensor
    observations: tuple[tuple[int, int], ...]


def run_workflow(
    *, theta: float = math.pi / 2, phi: float = 0.0, shots: int = 1000, seed: int = 3
) -> Workflow:
    """Read Bob after explicitly deferred private measurement.

    Examples:
        >>> workflow = run_workflow(theta=0.73, phi=-0.41)
        >>> sum(count for _, count in workflow.observations)
        1000
    """
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in (theta, phi)
    ):
        raise ValueError("theta and phi must be finite real angles")
    if (
        isinstance(shots, bool)
        or not isinstance(shots, int)
        or not 1 <= shots <= 100000
    ):
        raise ValueError("shots must be an integer in [1, 100000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    program = build_program(theta, phi)
    result = fq.run(program)
    readout = dict(program.probabilities(result, registers=("bob_qubit",)))
    probabilities = torch.tensor(
        [readout.get((i,), 0.0) for i in range(2)], dtype=torch.float64
    )
    draws = torch.multinomial(
        probabilities,
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=2)
    observations = tuple((i, int(counts[i])) for i in range(2) if int(counts[i]) > 0)
    bob = program.registers["bob_qubit"]
    rest = tuple(i for i in range(program.circuit.n_wires) if i not in bob)
    state = (
        result.statevector()
        .reshape((2,) * program.circuit.n_wires)
        .permute((*bob, *rest))
        .reshape(2, -1)
    )
    density = state @ state.conj().T
    return Workflow(program, density, observations)
