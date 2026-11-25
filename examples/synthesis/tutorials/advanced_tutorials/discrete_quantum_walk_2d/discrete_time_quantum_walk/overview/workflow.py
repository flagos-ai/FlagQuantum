"""Full-size ring and torus walks, retained coin environments and actual draws."""

from dataclasses import dataclass
from typing import Annotated, Literal

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import H, Output, QNum, S, allocate, control, drop, qfunc

from ..comparisons.circuit import reference_circuit as reference_torus
from ..main import main as torus
from ..physics import COIN_AMPLITUDES, POSITION_AMPLITUDES


@qfunc
def increment(position: QNum) -> None:
    position += 1


@qfunc
def decrement(position: QNum) -> None:
    position -= 1


@qfunc
def ring(t: int, x: Output[Annotated[QNum, 6]]) -> None:
    coin: Annotated[QNum, 1] = QNum()
    allocate(x)
    allocate(coin)
    x ^= 32
    H(coin[0])
    S(coin[0])
    for step in range(t):
        H(coin[0])
        control(coin == 1, lambda: increment(x))
        control(coin == 0, lambda: decrement(x))
    drop(coin)


@dataclass(frozen=True)
class Evolution:
    dimension: int
    steps: int
    program: synth.Program
    maximum_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class Observation:
    position: tuple[int, ...]
    count: int
    probability: float


def build_program(dimension: Literal[1, 2], steps: int) -> synth.Program:
    """Specialize the original step count; preserve all original position bits."""
    if type(steps) is not int or not 0 <= steps <= 20:
        raise ValueError("use an integer step count in 0..20")
    if type(dimension) is not int or dimension not in (1, 2):
        raise ValueError("dimension must be one or two")
    if dimension == 1:
        return synth.synthesize(
            ring, classical_inputs={"t": steps}, memory_limit_bytes=64 * 1024**2
        )
    if dimension != 2:
        raise ValueError("dimension must be one or two")
    # The retained convenience wrapper has a narrower evidence cap. Reuse the
    # same qfunc through a checked binding at the Notebook's original twenty steps.
    return synth.synthesize(
        torus,
        classical_inputs={
            "t": steps,
            "x_amplitudes": list(POSITION_AMPLITUDES),
            "y_amplitudes": list(POSITION_AMPLITUDES),
            "coin_amplitudes": list(COIN_AMPLITUDES),
        },
        memory_limit_bytes=64 * 1024**2,
    )


def reference_ring(steps: int) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    circuit.x(0)
    circuit.h(6)
    circuit.s(6)
    shift = torch.zeros((128, 128), dtype=torch.complex128)
    for coin in range(2):
        for position in range(64):
            shifted = (position + (1 if coin else -1)) % 64
            shift[coin * 64 + shifted, coin * 64 + position] = 1
    for _ in range(steps):
        circuit.h(6)
        circuit.any(6, *range(6), matrix=shift)
    return circuit


def verify(dimension: Literal[1, 2], steps: int) -> Evolution:
    """Include the full entangled coin; require only compiler scratch to be zero."""
    return verify_program(dimension, steps, build_program(dimension, steps))


def verify_program(
    dimension: Literal[1, 2], steps: int, program: synth.Program
) -> Evolution:
    """Check a supplied native Notebook program and its retained coin."""
    expected = (
        fq.run(reference_ring(steps) if dimension == 1 else reference_torus(steps))
        .statevector()
        .flatten()
    )
    actual = fq.run(program).statevector().reshape(expected.numel(), -1)
    # Native outputs are allocated before the retained hidden coin and compiler
    # work follows both. The reference axes are x, [y,] coin, in that same order.
    logical = tuple(wire for word in program.registers.values() for wire in word)
    if logical != tuple(range(6 if dimension == 1 else 10)):
        raise AssertionError("unexpected native position layout")
    error = float((actual[:, 0] - expected).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    if error > 1e-12 or dirty > 1e-22:
        raise AssertionError(f"walk state error {error}, dirty compiler work {dirty}")
    return Evolution(dimension, steps, program, error, dirty)


def sample(
    program: synth.Program, *, shots: int = 1000, seed: int = 1970
) -> tuple[Observation, ...]:
    """Draw actual position observations, marginalizing the retained coin."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    values = dict(
        program.probabilities(fq.run(program), registers=tuple(program.registers))
    )
    positions = tuple(values)
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(tuple(values.values()), dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(positions),
    )
    return tuple(
        Observation(position, int(counts[index]), int(counts[index]) / shots)
        for index, position in enumerate(positions)
    )


def run_workflow() -> tuple[tuple[Evolution, tuple[Observation, ...]], ...]:
    return tuple(
        (row, sample(row.program))
        for row in (verify(1, 1), verify(1, 20), verify(2, 20))
    )
