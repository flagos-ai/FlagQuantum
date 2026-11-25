"""Run the original-size local state-preparation exercises and references."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.synth import Program, synthesize

from . import quantum

OUTCOMES = 32
SHOTS = 2048
SEED = 5
AMPLITUDES = tuple((word / OUTCOMES) ** 2 for word in range(OUTCOMES))


@dataclass(frozen=True)
class Check:
    qubits: int
    operations: int
    depth: int
    state_error: float
    workspace_probability: float


@dataclass(frozen=True)
class Sampling:
    joint: tuple[tuple[tuple[float, int], int], ...]
    marginal: tuple[tuple[float, int], ...]
    expected_indicator_probability: float


def _check(program: Program, state: torch.Tensor, expected: torch.Tensor) -> Check:
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    state = state.reshape(expected.numel(), -1)
    error = float((state[:, 0] - expected).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    if error > 3e-12 or dirty > 1e-22:
        raise AssertionError(f"full state error {error}, workspace probability {dirty}")
    return Check(
        resources.n_qubits, resources.n_operations, resources.depth, error, dirty
    )


def arithmetic() -> Check:
    program = synthesize(quantum.arithmetic, max_qubits=14, max_workspace=2)
    expected = torch.zeros(4096, dtype=torch.complex128)
    for x in range(8):
        for z in range(8):
            y = 2 * x + 4 * z + 2
            expected[(x << 9) | (y << 3) | z] = 1 / 8
    return _check(program, fq.run(program).statevector(), expected)


def random_probabilities() -> tuple[float, ...]:
    from numpy.random import RandomState

    values = RandomState(0).rand(OUTCOMES)
    return tuple(float(value) for value in values / values.sum())


def prepare_random(probabilities: Sequence[float]) -> tuple[Check, float]:
    program = synthesize(
        quantum.random_distribution,
        classical_inputs={"probabilities": probabilities},
        max_qubits=5,
        max_workspace=0,
    )
    state = fq.run(program).statevector().flatten()
    reference = torch.tensor(probabilities, dtype=torch.float64)
    distance = float(torch.linalg.vector_norm(state.abs().square() - reference))
    assert distance <= 0.01
    return _check(program, state, reference.sqrt().to(torch.complex128)), distance


def signed_word(word: int, width: int) -> int:
    return word - 2**width if word & (1 << (width - 1)) else word


def _signed_sum() -> fq.Circuit:
    """Accumulate sign-extended y and twice x in raw half-unit words."""
    circuit = fq.Circuit(14, dtype=torch.complex128)
    y_wires = (7, 6, 5, 4, 3)
    result_wires = (13, 12, 11, 10, 9, 8)
    for source, target in zip((*y_wires, y_wires[-1]), result_wires, strict=True):
        circuit.cx(source, target)
    for i, wire in enumerate((2, 1, 0)):
        increment = 1 << (i + 1)
        permutation = torch.eye(128, dtype=torch.complex128)
        for word in range(64):
            destination = 64 + ((word + increment) % 64)
            permutation[destination, 64 + word] = 1
            permutation[64 + word, 64 + word] = 0
        circuit.any(wire, *result_wires[::-1], matrix=permutation)
    return circuit


def fixed_point() -> tuple[tuple[float, float, float], Check]:
    """Execute the source's initialized operands and check every coherent input."""
    compiled = _signed_sum()
    input_registers = fq.Circuit(14, dtype=torch.complex128)
    input_registers.x(0).x(2).x(3).x(7)
    input_registers.compose(compiled)
    state = input_registers.state().flatten()
    raw = int(state.abs().argmax())
    x = (raw >> 11) & 7
    y = (raw >> 6) & 31
    answer = raw & 63

    witness = fq.Circuit(14, dtype=torch.complex128)
    for wire in range(8):
        witness.h(wire)
    witness.s(0).compose(compiled)
    result = witness.state().flatten()
    reference = torch.zeros_like(result)
    for first in range(8):
        for second in range(32):
            result_word = (2 * first + signed_word(second, 5)) % 64
            index = (first << 11) | (second << 6) | result_word
            component = reference.imag if first & 4 else reference.real
            component[index] = 1 / 16
    error = float((result - reference).abs().max())
    assert error < 3e-12
    resources = estimate_resources(input_registers)
    return (
        (float(x), signed_word(y, 5) / 2, signed_word(answer, 6) / 2),
        Check(resources.n_qubits, resources.n_operations, resources.depth, error, 0.0),
    )


def amplitude_assignment() -> tuple[Check, Sampling]:
    angles = [2 * math.asin(amp) for amp in AMPLITUDES]
    program = synthesize(
        quantum.amplitude_assignment,
        classical_inputs={"angles": angles},
        max_qubits=6,
        max_workspace=0,
    )
    result = fq.run(program)
    reference = torch.tensor(
        [
            component / math.sqrt(OUTCOMES)
            for amp in AMPLITUDES
            for component in (math.sqrt(1 - amp**2), amp)
        ],
        dtype=torch.complex128,
    )
    checked = _check(program, result.statevector(), reference)
    probability = program.probabilities(result, registers=("x", "indicator"))
    words = tuple(probability)
    draws = torch.multinomial(
        torch.tensor([probability[word] for word in words], dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    counts = Counter((words[i][0] / OUTCOMES, words[i][1]) for i in draws.tolist())
    marginal: Counter[float] = Counter()
    for (x, _), count in counts.items():
        marginal[x] += count
    weighted_norm = math.fsum(a**2 for a in AMPLITUDES) / OUTCOMES
    return checked, Sampling(
        tuple(sorted(counts.items())), tuple(sorted(marginal.items())), weighted_norm
    )


def exponential_qft() -> tuple[Check, tuple[float, ...], int]:
    weights = tuple(math.exp(-0.1 * word) for word in range(OUTCOMES))
    normalized = torch.tensor(weights, dtype=torch.float64)
    normalized = normalized / normalized.sum()
    program = synthesize(
        quantum.exponential_transform,
        classical_inputs={"probabilities": normalized.tolist()},
        max_qubits=5,
        max_workspace=0,
    )
    result = fq.run(program)
    expected = torch.fft.ifft(normalized.sqrt().to(torch.complex128), norm="ortho")
    state = result.statevector().flatten()
    checked = _check(program, state, expected)
    probability = state.abs().square()
    sample = torch.multinomial(
        probability, 1, generator=torch.Generator().manual_seed(SEED)
    )
    return checked, tuple(float(value) for value in probability), int(sample.item())
