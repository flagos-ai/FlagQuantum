"""Execute native circle and hypercube walks at the original sizes."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.comparisons.circuit import (
    reference_circuit as circle_reference,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_circle.main import (
    main as circle,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_circle_balanced_coin.main import (
    main as balanced_circle,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube.comparisons.circuit import (
    reference_circuit as cube_reference,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube.main import (
    build_program as cube,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

CIRCLE_SIZE = 128
CIRCLE_STEPS = 50
CUBE_DIMENSION = 4
CUBE_STEPS = 4
SHOTS = 10_000
SEED = 198


@dataclass(frozen=True)
class Observation:
    positions: tuple[int, ...]
    counts: tuple[int, ...]
    exact: tuple[float, ...]
    maximum_error: float
    work_probability: float
    resources: ResourceEstimate

    def observed(self, position: int) -> float:
        return self.counts[self.positions.index(position)] / SHOTS


@dataclass(frozen=True)
class Walks:
    circle: Observation
    balanced: Observation
    cube: Observation
    classical_circle: tuple[int, ...]
    classical_cube: tuple[int, ...]


def observe(
    program: synth.Program, reference: fq.Circuit, *, signed: bool, seed: int
) -> Observation:
    result = fq.run(program)
    expected = fq.run(reference).statevector().flatten()
    actual = result.statevector().reshape(expected.numel(), -1)
    error = float((actual[:, 0] - expected).abs().max())
    work = float(actual[:, 1:].abs().square().sum())
    if error >= 1e-11 or work >= 1e-24:
        raise AssertionError(
            f"Walk reference error {error}, workspace probability {work}"
        )
    raw = dict(program.probabilities(result, registers=("x",)))
    size = CIRCLE_SIZE if signed else 1 << CUBE_DIMENSION
    exact = torch.tensor(
        [raw.get((word,), 0.0) for word in range(size)], dtype=torch.float64
    )
    draws = torch.multinomial(
        exact,
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=size)
    if signed:
        exact = exact.roll(size // 2)
        counts = counts.roll(size // 2)
    positions = range(-size // 2, size // 2) if signed else range(size)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("Expected a measured circuit resource report")
    return Observation(
        tuple(positions),
        tuple(int(value) for value in counts),
        tuple(float(value) for value in exact),
        error,
        work,
        resources,
    )


def circle_walk(*, balanced: bool) -> Observation:
    program = synth.synthesize(
        balanced_circle if balanced else circle,
        classical_inputs={"t": CIRCLE_STEPS},
        max_qubits=8,
        max_workspace=0,
        max_operations=20_000,
        memory_limit_bytes=16 * 1024**2,
    )
    reference = circle_reference(width=7, steps=CIRCLE_STEPS, balanced=balanced)
    return observe(program, reference, signed=True, seed=SEED + int(balanced))


def cube_walk() -> Observation:
    return observe(
        cube(CUBE_STEPS),
        cube_reference(CUBE_STEPS),
        signed=False,
        seed=SEED + 2,
    )


def classical_circle() -> tuple[int, ...]:
    """Start each of the 10,000 classical circle trials from zero."""
    rng = torch.Generator().manual_seed(SEED + 3)
    flips = torch.randint(0, 2, (SHOTS, CIRCLE_STEPS), generator=rng)
    positions = (1 - 2 * flips).sum(dim=1) % CIRCLE_SIZE
    decoded = torch.bincount(positions, minlength=CIRCLE_SIZE).roll(CIRCLE_SIZE // 2)
    return tuple(int(value) for value in decoded)


def classical_cube() -> tuple[int, ...]:
    """Retain the source's single trajectory and visit count at every move."""
    rng = torch.Generator().manual_seed(SEED + 4)
    position = 0
    visits = [0] * (1 << CUBE_DIMENSION)
    for direction in torch.randint(0, CUBE_DIMENSION, (SHOTS,), generator=rng):
        position ^= 1 << int(direction)
        visits[position] += 1
    return tuple(visits)


def run() -> Walks:
    return Walks(
        circle_walk(balanced=False),
        circle_walk(balanced=True),
        cube_walk(),
        classical_circle(),
        classical_cube(),
    )
